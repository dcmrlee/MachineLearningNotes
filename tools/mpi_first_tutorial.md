# MPI初体验
在容器内，体验MPI的效果


## 创建MPI镜像
1. 准备的环境：
    - CentOS 7.x
    - ssh: yum install openssh openssh-server openssh-clients
    - gcc: yum install gcc gcc-c++
    - make: yum install make
    - rpmbuild: yum install rpmbuild
    - glibc: yum install glibc-headers

2. 下载OpenMPI包
    例如：openmpi-3.1.0-1.src.rpm，注意这里是source rpm，而并非binary rpm

3. 安装MPI环境
    - rpm -ivh openmpi-3.1.0-1.src.rpm
    - cd rpmbuild/SPECS
    - rpmbuild -ba openmpi-3.1.0.spec
    - cd rpmbuild/RPMS/x86_64/
    - rpm -i openmpi-3.1.0-1.el7.centos.x86_64.rpm

4. 检查安装环境
    安装后应该可以看到有mpirun mpicc mpiexec等命令

## 编写MPI的"Hello World"
C的代码：mympi.c
```
#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    int provided, claimed;
    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
    MPI_Query_thread( &claimed );
    printf( "Query thread level= %d  Init_thread level= %d\n", claimed, provided );
    MPI_Finalize();
}
```

## 运行MPI程序

1. 编译
mpicc -o mympi mympi.cc
2. 运行
mpirun --allow-run-as-root -np 1 mympi
mpirun --allow-run-as-root -np 4 mpi
输出如下：
```
Query thread level= 3  Init_thread level= 3
```
或
```
Query thread level= 3  Init_thread level= 3
Query thread level= 3  Init_thread level= 3
Query thread level= 3  Init_thread level= 3
Query thread level= 3  Init_thread level= 3
```
