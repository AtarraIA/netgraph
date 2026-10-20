
/* ---------- boot ---------- */
if (!load()) openDefault();   // a first visit opens the example autoencoder, trained
if (canFetch()) dsLoading = {id:'mnist', name:'MNIST', frac:0};   // until the stored datasets are read, then MNIST is fetched if missing
renderPalette();
render();
flushSave();
   // the starting state is the first undo snapshot
requestAnimationFrame(fit);
loadDatasets().then(loadWeights).then(() => { render(); return autoFetch(); });
