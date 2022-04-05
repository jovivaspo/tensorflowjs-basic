//Caracterísitacas de un tensor:
/*Data Type (DType)
Shape
Rank / Axis
Size*/

/*Tipos de tensores
1D: Coordenadas x, y, z de un punto
2D: Imágenes en escala de grises
3D:  Imágenes en color RGB
4D: Vídeos
5D: Lote de vídeo o mineCraft
6D: Lote de animaciones
*/

//Crear un tensor de rango 2 => tensor 2D
const tensor2D = tf.tensor2d([[1,1,1],[1,1,1]])
tensor2D.print()
//Creamos un tensor1D
const tensor1D = tf.tensor1d([1, 2, 3, 4])
tensor1D.print()
//Crear un escalar
let scalar = tf.scalar(2).print();

//Modificamos el tensor de 1D a 2D
const newTensor2D = tensor1D.reshape([2, 2]).print();
//Modificamos el tensor de 2D a 1D
const newTensor1D = tensor2D.reshape([6]).print()


/*Carga de modelo existente mediante url*/
const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = undefined;


//Cargar un modelo
async function loadModel() {

  model = await tf.loadLayersModel(MODEL_PATH);

  model.summary(); ///Informa sobre las capas salidas y entradas del modelo

  

  // Create a batch of 1.

  const input = tf.tensor2d([[870]]);

  

  // Create a batch of 3

  const inputBatch = tf.tensor2d([[500], [1100], [970]]);


  // Actually make the predictions for each batch.

  const result = model.predict(input);

  const resultBatch = model.predict(inputBatch);

  

  // Print results to console.

  result.print();  // Or use .arraySync() to get results back as array.

  resultBatch.print(); // Or use .arraySync() to get results back as array.

  

  input.dispose();

  inputBatch.dispose();

  result.dispose();

  resultBatch.dispose();

  model.dispose();

}


loadModel();


///CARGANDO MODELOS PREENTRENADOS DE HUB
const MODEL_PATH_HUB = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
const EXAMPLE_IMG = document.getElementById('exampleImg');

let movenet = undefined;


async function loadAndRunModel() {

  movenet = await tf.loadGraphModel(MODEL_PATH_HUB, {fromTFHub: true});

  /*1º PRUEBA CREANDO UN TENSOR DE CEROS:*/

    //INPUT:
  let exampleInputTensor = tf.zeros([1, 192, 192, 3], 'int32'); //LOTE 1, ALTURA, ANCHO, Y RBG COLORS

  
//OUTPUT SERÁ UN VECTOR DE 17 ELEMENTOS, DONDE CADA ELEMENTO ES UN ARRAY CON 3 ELEMENTOS: X%, Y% Y LA PREDICCIÓN
  let tensorOutput = movenet.predict(exampleInputTensor);

  let arrayOutput = await tensorOutput.array();


  console.log(arrayOutput);

  //FIN DE LA PRUEBA//

  /*ESTUDIANDO LA IMAGEN DE LA PAGINA WEB*/

  /*lA IMAGEN NO POSEE LA ANCHURA REQUERIDA DE 192*192 POR LO QUE SE DEBE DE HACER UN RECORTE BUSCANDO A LA PERSONA*/

let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG); //LECTURA DE LA FOTO

  console.log(imageTensor.shape);  //ACTUALMENTE LA IMAGEN ES DE [1,360,640,3]

  //REALIZAMOS EL RECORTE, EN PRIMER LUGAR SE INDICA LA POSICIÓN INICIAL DEL RECORTE [15,170,0] 'X,Y,COLOR'


  let cropStartPoint = [15, 170, 0];

  let cropSize = [345, 345, 3]; //ALTURA Y ANCHURA DEL ÁREA DE RECORTE Y 'BLUE' COMO COLOR

  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize); //REALIZAMOS EL RECORTE


  let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt(); //FINALMENTE AJUSTAMOS EL RECORTE A 192*192

  console.log(resizedTensor.shape);

  

  let tensorOutputImage = movenet.predict(tf.expandDims(resizedTensor));

  let arrayOutputImage = await tensorOutputImage.array();

  console.log(arrayOutputImage);

  /*EL ELEMENTO 0 DEL ARRAY ES LA NARIZ, DONDE VEMOS QUE SE INDICA EN %SU POSICIÓN, X E Y, Y EL ULTIMO VALOR INDICA EL % DE PRECCIÓN O ACIERTO DE ESTO SEA LA NARIZ
  FINALMENTE TENEMOS 17 PUNTOS REFERENTES A 17 PUNTOS DIFERENTES DEL CUERPO DE LA PERSONA */
}


loadAndRunModel();