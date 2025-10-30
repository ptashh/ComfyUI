import threading
from simple_realistic_benchmark import benchmark_time, monitor_resources

def run_both():
    """Run timing benchmark and resource monitoring simultaneously"""
    
    # Storage for results
    time_results = {}
    resource_results = {}
    
    def run_benchmark():
        nonlocal time_results
        time_results = benchmark_time(10, 512, 512, 20)
    
    def run_monitor():
        nonlocal resource_results
        resource_results = monitor_resources(100) 
    
    # Start both threads
    monitor_thread = threading.Thread(target=run_monitor)
    benchmark_thread = threading.Thread(target=run_benchmark)
    
    print("Starting monitoring and benchmark...\n")
    monitor_thread.start()
    benchmark_thread.start()
    
    # Wait for both to finish
    benchmark_thread.join()
    monitor_thread.join()
    
    # Print results
    print("\n=== TIMING ===")
    print(f"Avg: {time_results['avg_time']:.2f}s")
    print(f"Min: {time_results['min_time']:.2f}s")
    print(f"Max: {time_results['max_time']:.2f}s")
    
    print("\n=== RESOURCES ===")
    print(f"CPU avg: {resource_results['cpu']['avg']:.1f}%")
    print(f"RAM avg: {resource_results['ram']['avg']:.1f}%")
    if resource_results['gpu']:
        print(f"GPU avg: {resource_results['gpu']['avg_usage']:.1f}%")
        print(f"GPU mem avg: {resource_results['gpu']['avg_memory']:.1f}%")

if __name__ == "__main__":
    run_both()
