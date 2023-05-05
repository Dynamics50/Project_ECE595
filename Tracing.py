from Operation import single_run
from memory_profiler import profile

@profile                                # Tracing the performance of PC 
def memory_Tracing():
    single_run(False)

if __name__ == "__main__":
    memory_Tracing()


