#include <iostream>
#include <assert.h>
#include <png.h>

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
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

//  printf("filename = %s\n", filename);


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

//  printf("readPngImage 1 == \n");

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

//  printf("readPngImage 2 == \n");

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

//  printf("readPngImage 3 == \n");

  auto *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (size_t y = 0; y < height; y++) {
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, info_ptr);

  printf("width = %zu height = %zu numChannels = %zu \n", width, height, numChannels);


  imageDims[0] = width;
  imageDims[1] = height;
  imageDims[2] = numChannels;
  imageT = static_cast<float *>(
      calloc(1, width * height * numChannels * sizeof(float)));

//  printf("readPngImage 4 == \n");

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

//  printf("readPngImage 5 == \n");

  for (size_t y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
  fclose(fp);
  printf("Loaded image: %s\n", filename);

  return false;
}


static void loadImagesAndPreprocess(char*filename, float *resultT, int64_t shape[]) {

  std::pair<float, float> range = std::make_pair(0., 1.0);
  unsigned numImages = 1;

  // N x C x H x W
//  resultDims[0] = numImages;
//  resultDims[1] = 3;
//  resultDims[2] = DEFAULT_HEIGHT;
//  resultDims[3] = DEFAULT_WIDTH;
//  printf("1------\n");
//  size_t resultSizeInBytes = 1;
//  for(int i = 0; i < 4; i++) {
//    resultSizeInBytes = resultSizeInBytes * shape[i];
//  }

//  printf("2------\n");
//  resultSizeInBytes = resultSizeInBytes * sizeof(float);

//  resultT = static_cast<float *>(malloc(resultSizeInBytes));
  // We iterate over all the png files, reading them all into our result tensor
  // for processing
//  for (unsigned n = 0; n < numImages; n++) {
    float *imageT{nullptr};
    size_t dims[3];
    bool loadSuccess = !readPngImage(filename, range, imageT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

//    printf("3------\n");

//    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) &&
    assert((dims[0] == shape[2] && dims[1] == shape[3]) &&
           "All images must have the same Height and Width");

    // Convert to BGR, as this is what NN is expecting.
    size_t idx = 0;
    for (unsigned z = 0; z < 1; z++) {
      for (unsigned y = 0; y < dims[1]; y++) {
        for (unsigned x = 0; x < dims[0]; x++) {
//          resultT[getXYZW(dims, 1, 0, x, y)] =
              resultT[idx++] =
              imageT[getXYZ(dims, x, y, z)];
        }
      }
    }
//  }


//  printf("Loaded images size in bytes is: %lu\n", resultSizeInBytes);
}


