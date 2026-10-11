
/* ---------- boot ---------- */
if (!load()) example();
renderPalette();
render();
flushSave();
   // the starting state is the first undo snapshot
requestAnimationFrame(fit);
loadDatasets().then(loadWeights).then(() => { render(); return autoFetch(); });
