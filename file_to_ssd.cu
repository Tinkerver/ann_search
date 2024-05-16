#include <cstdio>
#include <cstdlib>
#include <vector>
#include "lightbam.cuh"
const int max_io_size = 65536;
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s file_name ssd_offset\n", argv[0]);
        return 1;
    }
    FILE *fp = fopen(argv[1], "rb");
    size_t offset = atol(argv[2]);

    std::vector<Device *> devices{new Device(0)};
    Controller *ctrl = new ControllerDecoupled(devices, -1, max_io_size);
    void *buf = malloc(max_io_size);
    while (!feof(fp))
    {
        fread(buf, 1, max_io_size, fp);
        ctrl->write_data(offset / AEOLUS_LB_SIZE, max_io_size / AEOLUS_LB_SIZE, buf);
        offset += max_io_size;
    }
    free(buf);
    delete ctrl;
    delete devices[0];
    fclose(fp);
    return 0;
}