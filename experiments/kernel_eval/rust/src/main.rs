use cutile::prelude::*;
use half::bf16;
use std::sync::Arc;
use std::io::{self, BufRead};

mod kernels;
use kernels::kernels as gpu_kernels;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for line in io::stdin().lock().lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let bits = v.get("gate_bits").is_some();
        let read = |name: &str| -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
            let key = if bits { format!("{name}_bits") } else { name.to_string() };
            v[&key].as_array().ok_or("input array missing")?.iter().map(|x| {
                if bits {
                    let n = x.as_u64().filter(|n| *n <= 65535).ok_or("invalid BF16 bits")?;
                    Ok(bf16::from_bits(n as u16))
                } else {
                    Ok(bf16::from_f32(x.as_f64().ok_or("invalid finite number")? as f32))
                }
            }).collect()
        };
        let g = read("gate")?;
        let u = read("up")?;
        if g.is_empty() || g.len()!=u.len() || g.len()>1048576 { return Err("invalid lengths".into()); }
        let n = g.len();
        let mut g = api::copy_host_vec_to_device(&Arc::new(g)).sync_on(&stream)?;
        let u = api::copy_host_vec_to_device(&Arc::new(u)).sync_on(&stream)?;
        let mut out = api::zeros::<bf16>(&[n]).sync_on(&stream)?;
        gpu_kernels::gelu_mul(&g, &u, (&mut out).partition([256]))
            .generics(vec!["256".into()]).sync_on(&stream)?;
        let graph = cutile::cuda_async::cuda_graph::CudaGraph::scope(&stream, |scope| {
            for _ in 0..100 {
                scope.record(gpu_kernels::gelu_mul(&g, &u, (&mut out).partition([256]))
                    .generics(vec!["256".into()]))?;
            }
            Ok(())
        })?;
        graph.launch().sync_on(&stream)?;
        let mut timings = Vec::new();
        for _ in 0..5 {
            let start = stream.device().new_event()?;
            let end = Arc::new(stream.device().new_event()?);
            let end_op = end.clone();
            start.record(&stream)?;
            graph.launch().and_then_with_context(move |ctx, _| {
                value(end_op.record(ctx.get_cuda_stream()))
            }).sync_on(&stream)??;
            end.synchronize()?;
            timings.push(start.elapsed_time(&end)? as f64 * 1000.0 / 100.0);
        }
        let out = Arc::new(out);
        let host = (&out).to_host_vec().sync_on(&stream)?;
        let values: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
        gpu_kernels::change((&mut g).partition([256])).generics(vec!["256".into()]).sync_on(&stream)?;
        graph.launch().sync_on(&stream)?;
        let changed = (&out).to_host_vec().sync_on(&stream)?;
        let output_bits: Vec<u16> = host.iter().map(|x| x.to_bits()).collect();
        let changed_bits: Vec<u16> = changed.iter().map(|x| x.to_bits()).collect();
        let changed: Vec<f32> = changed.iter().map(|x| x.to_f32()).collect();
        if bits {
            println!("{}", serde_json::json!({"output_bits": output_bits, "changed_bits": changed_bits}));
            continue;
        }
        println!("{}", serde_json::json!({"output": values, "changed_output": changed, "graph_us_per_call": timings}));
    }
    Ok(())
}
