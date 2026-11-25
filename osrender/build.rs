fn main() {
    // src/kernels.ptx is generated from src/kernels.cu (see README); it is
    // committed so building the binary does not need the CUDA toolkit.
    println!("cargo:rerun-if-changed=src/kernels.ptx");
    println!("cargo:rerun-if-changed=src/burn_kernels.ptx");
}
