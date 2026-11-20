use cutile::prelude::*;
use half::bf16;
use std::sync::Arc;
use std::io::{self, BufRead};

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    pub fn gelu_mul<const BLOCK: i32>(
        g: &Tensor<bf16, {[-1]}>, u: &Tensor<bf16, {[-1]}>,
        out: &mut Tensor<bf16, {[BLOCK]}>,
    ) {
        let block = get_tile_block_id().0;
        let x: Tile<f32, {[BLOCK]}> = convert_tile(g.partition(shape![BLOCK]).load([block]));
        let y: Tile<f32, {[BLOCK]}> = convert_tile(u.partition(shape![BLOCK]).load([block]));
        let inner = (x + x*x*x*constant(0.044715f32, shape![BLOCK]))*constant(0.7978845608028654f32, shape![BLOCK]);
        let a = (x*constant(0.5f32, shape![BLOCK]))*(tanh(inner)+constant(1.0f32, shape![BLOCK]));
        let rounded: Tile<bf16, {[BLOCK]}> = convert_tile(a);
        let rounded: Tile<f32, {[BLOCK]}> = convert_tile(rounded);
        let result: Tile<bf16, {[BLOCK]}> = convert_tile(rounded*y);
        out.store(result);
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for line in io::stdin().lock().lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let g: Vec<bf16> = v["gate"].as_array().ok_or("gate missing")?.iter()
            .map(|x| x.as_f64().map(|x| bf16::from_f32(x as f32)).ok_or("invalid gate"))
            .collect::<Result<_,_>>()?;
        let u: Vec<bf16> = v["up"].as_array().ok_or("up missing")?.iter()
            .map(|x| x.as_f64().map(|x| bf16::from_f32(x as f32)).ok_or("invalid up"))
            .collect::<Result<_,_>>()?;
        if g.is_empty() || g.len()!=u.len() || g.len()>1048576 { return Err("invalid lengths".into()); }
        let n = g.len();
        let g = api::copy_host_vec_to_device(&Arc::new(g)).sync_on(&stream)?;
        let u = api::copy_host_vec_to_device(&Arc::new(u)).sync_on(&stream)?;
        let mut out = api::zeros::<bf16>(&[n]).sync_on(&stream)?;
        kernels::gelu_mul(&g, &u, (&mut out).partition([256]))
            .generics(vec!["256".into()]).sync_on(&stream)?;
        let host = out.to_host_vec().sync_on(&stream)?;
        let values: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
        println!("{}", serde_json::json!({"output": values}));
    }
    Ok(())
}
