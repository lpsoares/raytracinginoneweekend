//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================
// Portado para GPU por Luciano Soares <lpsoares@gmail.com>

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#include "curand_kernel.h"

#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"

__device__  vec3 color(const ray& r2, hitable_list *world) {

    hit_record rec;
    ray r = r2;
    vec3 paint(1,1,1);

    for(int f=0;f<50;f++) {

        if (world->hit(r, 0.001, MAXFLOAT, rec)) { 
            
            ray scattered;
            vec3 attenuation;

            if(rec.mat_ptr->type==0) {
                lambertian *l = (lambertian *)rec.mat_ptr;
                if(l->scatter(r, rec, attenuation, scattered)) {
                    paint *= attenuation;
                    r = scattered;
                }
            } else if(rec.mat_ptr->type==1) {
                metal *l = (metal *)rec.mat_ptr;
                if(l->scatter(r, rec, attenuation, scattered)) {
                    paint *= attenuation;
                    r = scattered;
                }
            } else if(rec.mat_ptr->type==2) {
                dielectric *l = (dielectric *)rec.mat_ptr;
                if(l->scatter(r, rec, attenuation, scattered)) {
                    paint *= attenuation;
                    r = scattered;
                }
            } 

        }
        else {
            vec3 unit_direction = unit_vector(r.direction());
            float t = ((float)0.5)*(unit_direction.y() + ((float)1.0));
            paint *= ((float)1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return(paint);
        }

    }

    return(paint);
}

__global__ void cor(float *mem, int nx, int ny, camera cam, hitable_list *world, int ns, int div, int ind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    if (i < nx && j < (ny/div)) {
        int pos = j*nx + i;

        vec3 col(0, 0, 0);
        for (int s=0; s < ns; s++) {
            float u = float(i + curand_uniform(&state)) / float(nx);
            float v = (float( (ind*ny/div) + j + curand_uniform(&state)) / float(ny));
            ray r = cam.get_ray(u, v);
            //vec3 p = r.point_at_parameter(2.0);
            col += color(r, world);
        }
        col /= float(ns);
        col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

        mem[(pos*3)+0] = col[0];
        mem[(pos*3)+1] = col[1];
        mem[(pos*3)+2] = col[2];

    }

}

//hitable *random_scene() {
hitable **random_scene(int *amostras) {

    int n = 500;
    hitable **list = new hitable*[n+1];
    
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));

    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48()); 
            if ((center-vec3(4,0.2,0)).length() > 0.9) { 
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    *amostras = i; // define o numero de amostras

    return list;
}


int main() {

    // int nx = 1200;
    // int ny = 800;
    // int ns = 10;
    
    int nx = 4096;
    int ny = 2160;
    int ns = 300;

    auto start = std::chrono::high_resolution_clock::now();

    // Criacao da Camera
    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);

    /*
    hitable **list = NULL;
    list = new hitable*[5];
    float R = cos(M_PI/4);
    list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
    list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
    list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
    list[3] = new sphere(vec3(-1,0,-1), 0.5, new dielectric(1.5));
    list[4] = new sphere(vec3(-1,0,-1), -0.45, new dielectric(1.5));
    */

    int devicecount;
    cudaGetDeviceCount(&devicecount);   // recupera o número de devices (GPUs) no sistema
    
    cudaStream_t strm[devicecount];
    hitable_list *world_h[devicecount];
    hitable_list *world_d[devicecount];
    float *col_h[devicecount];
    float *col_d[devicecount];

    int n;
    hitable **list = random_scene(&n);

    // Aloca a memória em cada GPU
    for (int i = 0; i < devicecount; ++i) {
        cudaSetDevice(i);
        cudaMalloc((void**)&col_d[i],nx*(ny/devicecount)*3*sizeof(float));
        cudaMalloc((void**)&world_d[i],sizeof(hitable_list));

        cudaMallocHost((void**)&col_h[i], nx*(ny/devicecount)*3*sizeof(float)); // necessário para multiGPU
        
        //world_h[i] = new hitable_list(list,5);
        world_h[i] = (hitable_list *)(new hitable_list(list,n));
        cudaStreamCreate(&(strm[i]));
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx-1)/threadsPerBlock.x + 1, ((ny/devicecount)-1)/threadsPerBlock.y + 1);

    // GPU usando streams
    for (int i = 0; i < devicecount; ++i) {
        cudaSetDevice(i);
        cudaMemcpyAsync(world_d[i],world_h[i],sizeof(hitable_list),cudaMemcpyHostToDevice,strm[i]);
        cor<<<numBlocks, threadsPerBlock, 0, strm[i]>>>(col_d[i], nx, ny, cam, world_d[i], ns, devicecount, devicecount-i-1);
        cudaError_t err = cudaGetLastError(); if ( err != cudaSuccess ) std::cerr << "Err: " << cudaGetErrorString(err) <<std::endl;
        cudaMemcpyAsync(col_h[i], col_d[i], nx*(ny/devicecount)*3*sizeof(float), cudaMemcpyDeviceToHost,strm[i]);
    }

    for (int i = 0; i < devicecount; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(strm[i]);
        cudaStreamDestroy(strm[i]);
        cudaFree(world_d[i]);
        cudaFree(col_d[i]);
        delete world_h[i];
    }

    // Gera imagem em ppm
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int k = 0; k < devicecount; k++) {
        for (int j = (ny/devicecount)-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                int pos = j*nx + i;
                int ir = int(255.99*col_h[k][(pos*3)+0]); 
                int ig = int(255.99*col_h[k][(pos*3)+1]); 
                int ib = int(255.99*col_h[k][(pos*3)+2]); 
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    for (int i = 0; i < devicecount; ++i) {
        cudaFreeHost(col_h[i]);
    }

    // Calcula o tempo que passou
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    
    std::cerr << devicecount << "xGPUs : ";
    std::cerr << nx << "x" << ny << "@" << ns;
    std::cerr << " em " << diff.count() << "s" << std::endl;

}