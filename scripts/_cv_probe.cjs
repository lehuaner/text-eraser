// Probe whether the vendored inline opencv.js works in Node and has all symbols we need.
const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
function start(cv) {
  const need = ['inpaint','INPAINT_TELEA','threshold','THRESH_OTSU','connectedComponentsWithStats',
    'distanceTransform','DIST_L2','morphologyEx','MORPH_CLOSE','MORPH_RECT','MORPH_ELLIPSE',
    'bitwise_or','bitwise_and','resize','getStructuringElement','boundingRect','findContours',
    'contourArea','cvtColor','COLOR_RGB2GRAY','threshold','THRESH_BINARY','RETR_EXTERNAL',
    'CHAIN_APPROX_SIMPLE','Mat','Scalar','Point','Size','connectedComponents','CV_8UC1','CV_32FC1','CV_32S','CV_8UC3'];
  const missing = need.filter(k => cv[k] === undefined);
  console.log('ALL_PRESENT:', missing.length === 0);
  console.log('MISSING:', JSON.stringify(missing));
}
const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
