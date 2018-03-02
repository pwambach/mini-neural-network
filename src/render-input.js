/* not used for now */

// renders a mnist number into a canvas
export default function renderInput(canvas, data, columns, rows) {
  const context = canvas.getContext('2d');
  
  // Define the image dimensions
  const width = columns;
  const height = rows;
  canvas.width = width;
  canvas.height = height;
 
  // Create an ImageData object
  const imageData = context.createImageData(width, height);

  let pixelIndex = 0;
 
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      // Set the pixel data
      imageData.data[pixelIndex*4] = data[pixelIndex]; // Red
      imageData.data[pixelIndex*4+1] = data[pixelIndex]; // Green
      imageData.data[pixelIndex*4+2] = data[pixelIndex]; // Blue
      imageData.data[pixelIndex*4+3] = 255; // Alpha

      pixelIndex++;
    }
  }

  context.putImageData(imageData, 0, 0);
}
