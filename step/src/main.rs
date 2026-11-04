//! Run a Kimi-K3 decode manifest (`scripts/gen_stage.py` without `--chunk`) teacher-forced
//! on one rank: B sequences fed their given tokens one decode step at a
//! time, every step's logits kept. One process per rank; a group's ranks
//! meet through an NCCL id in a file on a shared disk, so they may sit on
//! different nodes. A manifest whose paged states are dealt by position
//! over the group runs one replicated batch: every member takes the same
//! rows, and only what addresses a member's own positions differs (its
//! slots, the pad's for the positions it does not hold, its local lengths,
//! its page and line tables).
//!
//!   k3_step --manifest dcp-l4-tp8.json --weights <HF checkpoint dir> --gpu 0 \
//!       --rank 0 --world 8 --nccl-id <shared dir>/id --tokens ids.i64 --rows 4 --out <dir>
//!
//! `ids.i64` is the rows' tokens, [rows, steps] i64; step s feeds every row
//! its token s at position s. Every rank writes `next.r<rank>.i64` [steps,
//! rows]; rank 0 also writes `top.f64` [steps, rows, 3] (the top-1 token, its
//! logit and the runner-up's) and `logits.f32` [kept, rows, vocab] of every
//! `--every`-th step (s + 1 a multiple of it; default 1, every step). Without
//! `--world` (a group of one, the manifest's own oracle) no NCCL is joined.
//! `--dump <buffer>` (repeatable) has rank 0 also write `<buffer>.bin`, the
//! buffer's live rows after every step: a workspace the last layer leaves
//! behind, such as its routing (`topk_idx`).
//! A manifest of two layers (`--layers 2`: one KDA layer, one MoE layer) has
//! no paged state and takes no slots or lengths. `--graph` replays the
//! decode program's CUDA graph (one per row count) instead of launching it.
//!
//! `--ctx N,... --bench 16,32,48,64` measures instead: every row holds N
//! positions of whatever the pages hold (finite garbage: the attention reads
//! it, nobody checks it), every step feeds each row a fresh pseudo-random
//! token, and each row count's graph is replayed `--iters` times after
//! `--warmup`. Each step is timed from the token write to the readback of
//! `next_token`; `bench.r<rank>.json` holds every row count's median, p10 and
//! p90, plus the back-to-back time of `--iters` replays (fresh tokens each) with one sync,
//! started right after a synchronous step so the ranks start together.
//! `--profile R@N,...` brackets three replays at R rows and context N with
//! cuProfilerStart/Stop (nsys `--capture-range=cudaProfilerApi`).
//!
//! Teacher-forced, `--every 0` keeps no logits and reads none back, so the
//! step is timed as the bench times it (token write to `next_token`), over
//! real text: `step.r<rank>.json` holds the p10/p50/p90 of the steps from
//! `--warmup` on, and `--profile-at S` brackets steps S..S+3.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kern_manifest::types::Dim;
use kern_pool::Interleave;
use kern_runtime::{Capacity, Offer, Runtime, Topology};

struct Args {
    manifest: PathBuf,
    weights: PathBuf,
    gpu: usize,
    rank: u64,
    world: u64,
    nccl_id: Option<PathBuf>,
    tokens: PathBuf,
    rows: usize,
    every: usize,
    dump: Vec<String>,
    out: PathBuf,
    graph: bool,
    ctx: Vec<u64>,
    bench: Vec<usize>,
    warmup: usize,
    iters: usize,
    profile: Vec<(usize, u64)>,
    profile_at: Option<usize>,
}

fn args() -> anyhow::Result<Args> {
    let mut a = Args {
        manifest: PathBuf::new(),
        weights: PathBuf::new(),
        gpu: 0,
        rank: 0,
        world: 1,
        nccl_id: None,
        tokens: PathBuf::new(),
        rows: 1,
        every: 1,
        dump: Vec::new(),
        out: PathBuf::new(),
        graph: false,
        ctx: Vec::new(),
        bench: Vec::new(),
        warmup: 5,
        iters: 50,
        profile: Vec::new(),
        profile_at: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow::anyhow!("{k} takes a value"));
        match k.as_str() {
            "--manifest" => a.manifest = v()?.into(),
            "--weights" => a.weights = v()?.into(),
            "--gpu" => a.gpu = v()?.parse()?,
            "--rank" => a.rank = v()?.parse()?,
            "--world" => a.world = v()?.parse()?,
            "--nccl-id" => a.nccl_id = Some(v()?.into()),
            "--tokens" => a.tokens = v()?.into(),
            "--rows" => a.rows = v()?.parse()?,
            "--every" => a.every = v()?.parse()?,
            "--dump" => a.dump.push(v()?),
            "--out" => a.out = v()?.into(),
            "--graph" => a.graph = true,
            "--eager" => a.graph = false,
            "--ctx" => a.ctx = v()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--bench" => a.bench = v()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--warmup" => a.warmup = v()?.parse()?,
            "--iters" => a.iters = v()?.parse()?,
            "--profile-at" => a.profile_at = Some(v()?.parse()?),
            "--profile" => {
                a.profile = v()?
                    .split(',')
                    .map(|w| {
                        let (r, c) = w.split_once('@').ok_or_else(|| anyhow::anyhow!("--profile takes rows@ctx,..."))?;
                        Ok((r.parse()?, c.parse()?))
                    })
                    .collect::<anyhow::Result<_>>()?
            }
            _ => anyhow::bail!("unknown arg {k}"),
        }
    }
    Ok(a)
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

/// A row's top-1 token, its logit and the runner-up's logit.
fn top2(row: &[f32]) -> [f64; 3] {
    let (i, v) = row.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
    let second = row.iter().enumerate().filter(|&(j, _)| j != i).fold(f32::MIN, |m, (_, &x)| m.max(x));
    [i as f64, v as f64, second as f64]
}

/// Rank 0 writes a fresh id; the others wait for it to appear.
/// Every member's offer (its manifest's digest, its handles, member 0's NCCL id), through files beside `path`.
fn offers(path: &Path, rank: u64, world: u64, own: &Offer) -> anyhow::Result<Vec<Offer>> {
    let file = |r: u64| path.with_extension(format!("o{r}"));
    let tmp = path.with_extension(format!("o{rank}-tmp"));
    std::fs::write(&tmp, own.to_bytes()?)?;
    std::fs::rename(&tmp, file(rank))?;
    let t = Instant::now();
    (0..world)
        .map(|r| loop {
            if let Some(o) = std::fs::read(file(r)).ok().and_then(|b| Offer::from_bytes(&b).ok()) {
                return Ok(o);
            }
            anyhow::ensure!(t.elapsed() < Duration::from_secs(3600), "no offer from rank {r} after an hour");
            std::thread::sleep(Duration::from_millis(200));
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a = args()?;
    let m = kern_manifest::Verified::from_json(&std::fs::read_to_string(&a.manifest)?)?;
    let benching = !a.ctx.is_empty();
    let ids: Vec<i64> = if benching {
        Vec::new()
    } else {
        std::fs::read(&a.tokens)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let (b, steps) = if benching {
        (a.bench.iter().copied().max().ok_or_else(|| anyhow::anyhow!("--ctx needs --bench <rows,...>"))?, 1)
    } else {
        (a.rows, ids.len() / a.rows)
    };
    anyhow::ensure!(b * steps == ids.len() || benching, "{} tokens are not {b} rows", ids.len());
    anyhow::ensure!(steps > 0, "no steps");
    let seqs_max = m.vars["seqs"].max as usize;
    anyhow::ensure!(b <= seqs_max, "{b} rows, the manifest takes at most {seqs_max}");
    let dealt = m.states.values().any(|s| s.shard.is_some());
    // a manifest with no MLA layer (one MoE layer) has no paged state to address
    let paged = m.buffers.contains_key("slot_mapping");
    anyhow::ensure!(dealt == (a.world > 1), "a dealt manifest runs over --world > 1, an undealt one alone");
    let il = Interleave::new(a.world)?;
    // a bench row also writes its next position, past the context
    let local = if benching { (a.ctx.iter().max().unwrap() + 1).div_ceil(a.world) as usize } else { steps.div_ceil(a.world as usize) };

    let t0 = Instant::now();
    let topo = Topology::one("tp", a.rank, a.world);
    // every row's local positions, plus the pad's page
    let capacity = Capacity { tokens: Some(((local + 64) * (b + 1)) as u64), slots: (b + 1) as u64, landings: 0 };
    let mut rt = Runtime::load(&m, None, a.gpu, Some(capacity), Some(&topo))?;
    rt.set_eager(!a.graph && !benching);
    rt.load_weights(&kern_runtime::Safetensors::open(&[&a.weights])?)?;
    if a.world > 1 {
        let path = a.nccl_id.as_ref().ok_or_else(|| anyhow::anyhow!("--world > 1 needs --nccl-id"))?;
        let members = offers(path, a.rank, a.world, &rt.offer()?)?;
        rt.connect(&BTreeMap::from([("tp".to_string(), members)]))?;
    }
    let once = BTreeMap::from([("tokens".to_string(), 1u64), ("seqs".to_string(), 1)]);
    for (name, p) in &m.programs {
        if p.once {
            rt.run(name, &once)?;
        }
    }
    eprintln!("rank {} loaded `{}` in {:.1} s", a.rank, m.model, t0.elapsed().as_secs_f64());

    let leases = (0..b).map(|_| if paged { rt.lease(local) } else { rt.lease_slot() }).collect::<Result<Vec<_>, _>>()?;
    // Positions this member does not hold write into a page nobody reads.
    let pad = rt.lease(1)?;
    let me = a.rank;
    let vars = BTreeMap::from([("tokens".to_string(), b as u64), ("seqs".to_string(), b as u64)]);
    for name in rt.page_tables().map(str::to_string).collect::<Vec<_>>() {
        let mut t = Vec::new();
        for l in &leases {
            l.extend_row(&name, &mut t)?;
        }
        rt.write_input_at(&name, &le(&t, i32::to_le_bytes), &vars)?;
    }
    for name in rt.seq_tables().map(str::to_string).collect::<Vec<_>>() {
        let lines = leases[0].seq_lines(&name)?;
        let mut t = Vec::with_capacity(lines * seqs_max);
        for r in 0..lines {
            for l in &leases {
                t.push(l.seq_line(&name, r)?);
            }
            t.resize((r + 1) * seqs_max, 0);
        }
        rt.write_input(&name, &le(&t, i32::to_le_bytes))?;
    }
    // the attention's query offsets (an fp8-cache manifest's), every sequence one row
    let cu = m.buffers.contains_key("cu_seqlens_q").then_some(seqs_max);
    if benching {
        return bench(&mut rt, &a, &leases, &pad, il, paged, cu);
    }
    if let Some(n) = cu {
        rt.write_input("cu_seqlens_q", &le(&cu_seqlens(b, n), i32::to_le_bytes))?;
    }

    let vocab = match m.buffers["logits"].shape[1] {
        Dim::Const(n) => n as usize,
        ref d => anyhow::bail!("logits width {d:?}"),
    };
    let row_bytes = |name: &str| -> anyhow::Result<usize> {
        let buf = m.buffers.get(name).ok_or_else(|| anyhow::anyhow!("no buffer `{name}` to dump"))?;
        let consts: Vec<u64> = buf.shape[1..].iter().map(|d| match d { Dim::Const(n) => Ok(*n), d => Err(anyhow::anyhow!("`{name}` dim {d:?}")) }).collect::<anyhow::Result<_>>()?;
        Ok((consts.iter().product::<u64>() * buf.dtype.bytes()) as usize)
    };
    let dumps: Vec<(String, usize)> = a.dump.iter().map(|n| Ok((n.clone(), row_bytes(n)? * b))).collect::<anyhow::Result<_>>()?;
    let mut dumped = vec![Vec::new(); dumps.len()];
    let (mut next, mut logits, mut top) = (Vec::new(), Vec::new(), Vec::<f64>::new());
    let t1 = Instant::now();
    let mut times = Vec::new();
    for s in 0..steps {
        let p = s as u64;
        if a.profile_at == Some(s) {
            unsafe { cudarc::driver::sys::cuProfilerStart() };
        }
        let t = Instant::now();
        let tok: Vec<i64> = (0..b).map(|r| ids[r * steps + s]).collect();
        rt.write_input_at("token_ids", &le(&tok, i64::to_le_bytes), &vars)?;
        if paged {
            let slot: Vec<i64> = leases
                .iter()
                .map(|l| if il.owner(p) == me { l.slot(il.local(p) as usize) } else { pad.slot(0) })
                .collect();
            let len = vec![il.len(p + 1, me) as i32; b];
            rt.write_input_at("slot_mapping", &le(&slot, i64::to_le_bytes), &vars)?;
            rt.write_input_at("seq_lens", &le(&len, i32::to_le_bytes), &vars)?;
        }
        rt.issue("decode", &vars)?;
        rt.synchronize()?;
        next.extend_from_slice(&rt.read_output("next_token")?[..b * 8]);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        if a.profile_at.map(|p| p + 3) == Some(s + 1) {
            unsafe { cudarc::driver::sys::cuProfilerStop() };
        }
        if me == 0 && a.every > 0 {
            let l = rt.read_buffer_prefix("logits", b * vocab * 4)?;
            let f: Vec<f32> = l.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            top.extend(f.chunks_exact(vocab).flat_map(top2));
            if (s + 1) % a.every == 0 {
                logits.extend(l);
            }
        }
        if me == 0 {
            for ((name, n), d) in dumps.iter().zip(&mut dumped) {
                d.extend(rt.read_buffer_prefix(name, *n)?);
            }
        }
    }
    let ms = t1.elapsed().as_secs_f64() * 1e3 / steps as f64;
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(a.out.join(format!("next.r{me}.i64")), &next)?;
    if me == 0 {
        std::fs::write(a.out.join("logits.f32"), &logits)?;
        std::fs::write(a.out.join("top.f64"), le(&top, f64::to_le_bytes))?;
        for ((name, _), d) in dumps.iter().zip(&dumped) {
            std::fs::write(a.out.join(format!("{name}.bin")), d)?;
        }
    }
    let how = if a.graph { "graph" } else { "eager" };
    println!("rank {me}: {steps} steps x {b} rows, {ms:.1} ms per step ({how}, with readback)");
    if a.every == 0 {
        let mut t: Vec<f64> = times[a.warmup.min(steps - 1)..].to_vec();
        t.sort_by(f64::total_cmp);
        let (p10, p50, p90) = (quantile(&t, 0.1), quantile(&t, 0.5), quantile(&t, 0.9));
        println!("rank {me} teacher-forced rows {b}: step p10/p50/p90 {p10:.2}/{p50:.2}/{p90:.2} ms");
        std::fs::write(
            a.out.join(format!("step.r{me}.json")),
            format!("{{\"rows\": {b}, \"steps\": {steps}, \"p10\": {p10:.3}, \"p50\": {p50:.3}, \"p90\": {p90:.3}}}\n"),
        )?;
    }
    Ok(())
}

/// A token in [1000, 101000) for step `s`, row `r`: distinct rows, no specials.
fn token(s: usize, r: usize) -> i64 {
    let mut x = (s as u64) << 32 | r as u64;
    x = (x ^ (x >> 31)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    1000 + ((x ^ (x >> 32)) % 100_000) as i64
}

/// `cu_seqlens_q` for `rows` sequences of one row each, over a buffer of `seqs_max + 1`.
fn cu_seqlens(rows: usize, seqs_max: usize) -> Vec<i32> {
    (0..=seqs_max).map(|i| i.min(rows) as i32).collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn bench(
    rt: &mut Runtime,
    a: &Args,
    leases: &[kern_pool::Lease],
    pad: &kern_pool::Lease,
    il: Interleave,
    paged: bool,
    cu: Option<usize>,
) -> anyhow::Result<()> {
    let me = a.rank;
    let mut report = Vec::new();
    let mut step = 0;
    for (&p, &rows) in a.ctx.iter().flat_map(|c| a.bench.iter().map(move |r| (c, r))) {
        let vars = BTreeMap::from([("tokens".to_string(), rows as u64), ("seqs".to_string(), rows as u64)]);
        if let Some(n) = cu {
            rt.write_input("cu_seqlens_q", &le(&cu_seqlens(rows, n), i32::to_le_bytes))?;
        }
        if paged {
            let slot: Vec<i64> = leases[..rows]
                .iter()
                .map(|l| if il.owner(p) == me { l.slot(il.local(p) as usize) } else { pad.slot(0) })
                .collect();
            rt.write_input_at("slot_mapping", &le(&slot, i64::to_le_bytes), &vars)?;
            rt.write_input_at("seq_lens", &le(&vec![il.len(p + 1, me) as i32; rows], i32::to_le_bytes), &vars)?;
        }
        let mut times = Vec::new();
        let mut finite = true;
        for i in 0..a.warmup + a.iters {
            let profiled = a.profile.contains(&(rows, p)) && i >= a.warmup && i < a.warmup + 3;
            if profiled && i == a.warmup {
                unsafe { cudarc::driver::sys::cuProfilerStart() };
            }
            let tok: Vec<i64> = (0..rows).map(|r| token(step, r)).collect();
            step += 1;
            let t = Instant::now();
            rt.write_input_at("token_ids", &le(&tok, i64::to_le_bytes), &vars)?;
            rt.issue("decode", &vars)?;
            rt.synchronize()?;
            let next = rt.read_output("next_token")?;
            let dt = t.elapsed().as_secs_f64() * 1e3;
            if profiled && i == a.warmup + 2 {
                unsafe { cudarc::driver::sys::cuProfilerStop() };
            }
            if i >= a.warmup {
                times.push(dt);
            }
            if i + 1 == a.warmup + a.iters {
                let l = rt.read_buffer_prefix("logits", rows * 4 * 1024)?;
                finite = l.chunks_exact(4).all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite());
                let n: Vec<i64> = next[..rows * 8].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                let picks: std::collections::BTreeSet<i32> = rt
                    .read_buffer_prefix("topk_idx", rows * 16 * 4)?
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let own = picks.iter().filter(|&&e| e as u64 % a.world == me).count();
                eprintln!("rank {me} rows {rows}: next {:?}, last layer's experts {} ({own} here)", &n[..rows.min(8)], picks.len());
            }
        }
        rt.issue("decode", &vars)?;
        rt.synchronize()?;
        let t = Instant::now();
        for _ in 0..a.iters {
            let tok: Vec<i64> = (0..rows).map(|r| token(step, r)).collect();
            step += 1;
            rt.write_input_at("token_ids", &le(&tok, i64::to_le_bytes), &vars)?;
            rt.issue("decode", &vars)?;
        }
        rt.synchronize()?;
        let b2b = t.elapsed().as_secs_f64() * 1e3 / a.iters as f64;
        times.sort_by(f64::total_cmp);
        let (p10, p50, p90) = (quantile(&times, 0.1), quantile(&times, 0.5), quantile(&times, 0.9));
        println!("rank {me} ctx {p} rows {rows}: step p10/p50/p90 {p10:.2}/{p50:.2}/{p90:.2} ms, back-to-back {b2b:.2} ms, logits finite {finite}");
        report.push(format!(
            "{{\"rows\": {rows}, \"ctx\": {p}, \"p10\": {p10:.3}, \"p50\": {p50:.3}, \"p90\": {p90:.3}, \"b2b\": {b2b:.3}, \"finite\": {finite}}}"
        ));
    }
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(a.out.join(format!("bench.r{me}.json")), format!("[{}]\n", report.join(",\n ")))?;
    Ok(())
}
