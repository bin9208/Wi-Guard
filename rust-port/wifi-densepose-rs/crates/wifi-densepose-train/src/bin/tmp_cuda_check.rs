use torch::{Device, Tensor};

fn main() {
    println!("TCH-RS CUDA Sanity Check");
    println!("CUDA available: {}", torch::Cuda::is_available());
    println!("CUDNN available: {}", torch::Cuda::cudnn_is_available());
    println!("Device count: {}", torch::Cuda::device_count());
    
    let device = if torch::Cuda::is_available() {
        Device::Cuda(1) // 4070
    } else {
        println!("CUDA NOT AVAILABLE, falling back to CPU");
        Device::Cpu
    };
    
    println!("Using device: {:?}", device);
    
    let t = Tensor::randn(&[3, 3], (torch::Kind::Float, device));
    t.print();
    println!("Tensor operation successful!");
}
