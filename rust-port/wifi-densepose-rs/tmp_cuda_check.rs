use tch;

fn main() {
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    if tch::Cuda::is_available() {
        for i in 0..tch::Cuda::device_count() {
            println!("Device {}: {}", i, tch::Cuda::get_device_name(i as i64));
        }
    }
}
