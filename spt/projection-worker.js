// Keep the legacy projection engine (including numeric's generated math functions)
// in a dedicated worker; the main page retains its strict script policy.
globalThis.window = globalThis;
globalThis.requestAnimationFrame = callback => setTimeout(callback, 0);
globalThis.onmessage = async event => {
  const {payload, method} = event.data;
  try {
    const {projectEmbedding} = await import('./projection-engine.js?v=70c86da32cf1ef32');
    const points = await projectEmbedding(payload, method, progress => postMessage({kind:'progress', ...progress}));
    postMessage({kind:'complete', points});
  } catch (err) { postMessage({kind:'error', message:err.message}); }
};
