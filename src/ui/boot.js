
/* ---------- boot ---------- */
if (!load()) openDefault();   // a first visit opens the example autoencoder, trained
renderPalette();
render();
flushSave();
   // the starting state is the first undo snapshot
requestAnimationFrame(fit);
loadDatasets().then(loadWeights).then(() => { render(); return autoFetch(); });
