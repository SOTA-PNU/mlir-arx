#include <iostream>
#include <cassert>
#include <png.h>
#include <cstring>

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224

void transpose(float* &in, float* &out) {

    float (*inPtr)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH];
    float (*outPtr)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3];

    inPtr = (float (*)[1][3][DEFAULT_HEIGHT][DEFAULT_WIDTH])in;
    outPtr = (float (*)[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3])out;

    for (int j=0; j<3; j++){
        for (int k=0; k<DEFAULT_HEIGHT; k++){
            for (int l=0; l<DEFAULT_WIDTH; l++){
                (*outPtr)[0][k][l][j] = (*inPtr)[0][j][k][l];
            }
        }
    }
}


/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const int64_t* dims, size_t x, size_t y, size_t z, size_t w) {
    return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
           (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) {
    return (x * dims[1] * dims[2]) + (y * dims[2]) + z;
}

/// Reads a PNG image from a file into a newly allocated memory block \p imageT
/// representing a WxHxNxC tensor and returns it. The client is responsible for
/// freeing the memory block.
bool readPngImage(const char *filename, std::pair<float, float> range,
                  float *&imageT, size_t *imageDims) {
    unsigned char header[8];
    // open file and test for it being a png.
    FILE *fp = fopen(filename, "rb");
    // Can't open the file.
    if (!fp) {
        return true;
    }

    // Validate signature.
    size_t fread_ret = fread(header, 1, 8, fp);
    if (fread_ret != 8) {
        return true;
    }
    if (png_sig_cmp(header, 0, 8)) {
        return true;
    }

    // Initialize stuff.
    png_structp png_ptr =
            png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        return true;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        return true;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        return true;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    size_t width = png_get_image_width(png_ptr, info_ptr);
    size_t height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
    const size_t numChannels = isGray ? 1 : 3;

    (void)bit_depth;
    assert(bit_depth == 8 && "Invalid image");
    assert((color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
            color_type == PNG_COLOR_TYPE_RGB || isGray) &&
           "Invalid image");
    bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

    int number_of_passes = png_set_interlace_handling(png_ptr);
    (void)number_of_passes;
    assert(number_of_passes == 1 && "Invalid image");

    png_read_update_info(png_ptr, info_ptr);

    // Error during image read.
    if (setjmp(png_jmpbuf(png_ptr))) {
        return true;
    }

    auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (size_t y = 0; y < height; y++) {
        row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, info_ptr);

    imageDims[0] = width;
    imageDims[1] = height;
    imageDims[2] = numChannels;
    imageT = static_cast<float *>(
            calloc(1, width * height * numChannels * sizeof(float)));

    float scale = ((range.second - range.first) / 255.0);
    float bias = range.first;

    for (size_t row_n = 0; row_n < height; row_n++) {
        png_byte *row = row_pointers[row_n];
        for (size_t col_n = 0; col_n < width; col_n++) {
            png_byte *ptr =
                    &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
            if (isGray) {
                imageT[getXYZ(imageDims, row_n, col_n, 0)] =
                        float(ptr[0]) * scale + bias;
            } else {
                imageT[getXYZ(imageDims, row_n, col_n, 2)] =
                        float(ptr[0]) * scale + bias;
                imageT[getXYZ(imageDims, row_n, col_n, 1)] =
                        float(ptr[1]) * scale + bias;
                imageT[getXYZ(imageDims, row_n, col_n, 0)] =
                        float(ptr[2]) * scale + bias;

            }
        }
    }

    for (size_t y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(fp);
    printf("Loaded image: %s\n", filename);

    return false;
}

//#define NHWC 0
//#define RANGE 255.0
#define RANGE 1.0

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(const char* filename, float* resultT, int64_t* shape) {
    //static void loadImagesAndPreprocess(char*filename, float *resultT) {
    //  assert(filenames.size() > 0 &&
    //         "There must be at least one filename in filenames");

    //for normalization range setting

    //for mxnet_exported_resnet18.onnx
    std::pair<float, float> range = std::make_pair(0., RANGE);

    //  size_t resultDims[4];
    unsigned numImages = 1;
    //  unsigned numImages = filenames.size();
    // N x C x H x W
    //  resultDims[0] = numImages;
    //  resultDims[0] = numImages;
    //  resultDims[3] = 3;
    //  resultDims[1] = DEFAULT_HEIGHT;
    //  resultDims[2] = DEFAULT_WIDTH;
    //  size_t resultSizeInBytes =
    //      numImages * 3 * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(float);
    //  resultT = static_cast<float *>(malloc(resultSizeInBytes));
    // We iterate over all the png files, reading them all into our result tensor
    // for processing
    //  for (unsigned n = 0; n < numImages; n++) {
    float *imageT{nullptr};
    size_t dims[3];
    //    bool loadSuccess = !readPngImage(filenames[n].c_str(), range, imageT, dims);
    bool loadSuccess = !readPngImage(filename, range, imageT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) &&
           "All images must have the same Height and Width");

    // Convert to BGR, as this is what NN is expecting.
    for (unsigned z = 0; z < 3; z++) {
        for (unsigned y = 0; y < dims[1]; y++) {
            for (unsigned x = 0; x < dims[0]; x++) {
                resultT[getXYZW(shape, 0, 2-z, x, y)] =
                        imageT[getXYZ(dims, x, y, z)];
            }
        }
    }
    //  }
    //  printf("Loaded images size in bytes is: %lu\n", resultSizeInBytes);

    //  if(NHWC == 1) {
    //    size_t bytes = sizeof(float[1][DEFAULT_HEIGHT][DEFAULT_WIDTH][3]);
    //    float* tmp = (float *)malloc(bytes);
    //    memset(tmp, 0, bytes);
    //    transpose(resultT, tmp);
    //    memcpy(resultT, tmp, bytes);
    //    free(tmp);
    //  }

}

