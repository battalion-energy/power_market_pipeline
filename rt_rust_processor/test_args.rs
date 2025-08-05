fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("Number of args: {}", args.len());
    for (i, arg) in args.iter().enumerate() {
        println!("  arg[{}] = '{}'", i, arg);
    }
    
    if args.len() > 1 && args[1] == "--bess-daily-revenue" {
        println!("✅ Found --bess-daily-revenue argument!");
    } else {
        println!("❌ Did not find --bess-daily-revenue argument");
    }
}