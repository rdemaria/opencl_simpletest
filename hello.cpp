#include <iostream>
#include <vector>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define SQ(f) (f)*(f)

// Compute c = a + b.
static const char source[] =
    "#if defined(cl_khr_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
    "#else\n"
    "#  error double precision is not supported\n"
    "#endif\n"
    "kernel void add(\n"
    "       ulong n,\n"
    "       global const double *a,\n"
    "       global const double *b,\n"
    "       global double *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "       c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";


int mk_test(std::vector<cl::Device> devices, int ndev,  cl::Context context) {
        //std::cout << "Using " << devices[ndev].getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create command queue.
        cl::CommandQueue queue(context, devices[ndev]);

        // Compile OpenCL program for found devices.
        cl::Program program(context, cl::Program::Sources(
                    1, std::make_pair(source, strlen(source))
                    ));

        try {
            program.build(devices);
        } catch (const cl::Error&) {
            std::cerr
                << "OpenCL compilation error" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[ndev])
                << std::endl;
            return 1;
        }

        cl::Kernel add(program, "add");
        size_t N = 1 << 20;

        // Prepare input data.
        std::vector<double> a(N, 0);
        std::vector<double> b(N, 0);
        std::vector<double> c(N);

        for (int jj=0; jj<N; jj++){
          a[jj]=1.0*jj;
          b[jj]=2.0*jj;
        };


        // Allocate device buffers and transfer input data to device.
        cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                a.size() * sizeof(double), a.data());

        cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                b.size() * sizeof(double), b.data());

        cl::Buffer C(context, CL_MEM_READ_WRITE,
                c.size() * sizeof(double));

        // Set kernel parameters.
        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        // Launch kernel on the compute device.
        queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);

        // Get result back to host.
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());

        // Should get '3' here.
        double err=0;
        for (int jj=0; jj<N; jj++){
          err+=SQ(a[jj]+b[jj]-c[jj]);
        };
        std::cout << "Difference C - OpenCL = " << err << std::endl;

};

int main(int argc, char *argv[]) {
    const size_t N = 1 << 20;

    try {
        // Get list of OpenCL platforms.
        std::vector<cl::Platform> platform;
        cl::Platform::get(&platform);

        if (platform.empty()) {
            std::cerr << "OpenCL platforms not found." << std::endl;
            return 1;
        }

        // Get all available devices.
        std::vector<cl::Device> devices;
        for(auto p = platform.begin(); devices.empty() && p != platform.end(); p++) {
            std::vector<cl::Device> pldev;
            try {
                p->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
                for(auto d = pldev.begin(); d != pldev.end(); d++) {
                    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
                    devices.push_back(*d);
                }
            } catch(...) {
                devices.clear();
            }
        }

        if (devices.empty()) {
            std::cerr << "GPUs with double precision not found." << std::endl;
            return 1;
        }

        // Create context
        cl::Context context;
        context = cl::Context(devices);

        std::cout << "Device list" << std::endl;
        for(int jj=0; jj<devices.size(); jj++){
           std::cout << jj<<":"<<devices[jj].getInfo<CL_DEVICE_NAME>() << std::endl;
           mk_test(devices,jj,context);
        };
	return 0;

    } catch (const cl::Error &err) {
        std::cerr
            << "OpenCL error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
        return 1;
    }
}
