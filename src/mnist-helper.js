const URLS = {
  train: {
    images: 'http://localhost:1235/train-images-idx3-ubyte',
    labels: 'http://localhost:1235/train-labels-idx1-ubyte'
  },
  test: {
    images: 'http://localhost:1235/t10k-images-idx3-ubyte',
    labels: 'http://localhost:1235/t10k-labels-idx1-ubyte'
  }
}

export default async function loadMNIST(type = 'train', start, length) {
  const dataFileBuffer = await fetchBinary(URLS[type].images);
  const labelFileBuffer = await fetchBinary(URLS[type].labels);
  const pixelValues = [];

  for (let index = start; index < start + length; index++) {
    const pixels = [];

    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        pixels.push(dataFileBuffer[(index * 28 * 28) + (x + (y * 28)) + 16]);
      }
    }

    const number = JSON.stringify(labelFileBuffer[index + 8]);
    const targets = Array.from({length: 10}).map((x, i) =>
      i === parseInt(number) ? 1 : 0
    );

    const imageData  = {
      number,
      targets,
      pixels
    };

    pixelValues.push(imageData);
  }

  return pixelValues;
}

function fetchBinary(url) {
  return fetch(url)
    .then(res => res.arrayBuffer())
    .then(buffer => new Uint8Array(buffer));
}






