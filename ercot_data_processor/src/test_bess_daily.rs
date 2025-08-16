use anyhow::Result;

fn main() -> Result<()> {
    println!("Testing BESS daily revenue processor directly...");
    
    // Call the processor directly
    super::bess_daily_revenue_processor::process_bess_daily_revenues()?;
    
    Ok(())
}