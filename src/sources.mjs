// The source files, in load order. The engine has no DOM (it also runs in the training worker and in the
// tests); the UI builds on it. build.mjs joins them into the single HTML file; the tests load the engine part.
export const ENGINE = ['model', 'data', 'shapes', 'expr', 'weights', 'run', 'export', 'project', 'worker'].map(f => `engine/${f}.js`);
export const UI = ['state', 'storage', 'loss-ui', 'pictures', 'canvas', 'inspector', 'neuro', 'app', 'train', 'events', 'boot'].map(f => `ui/${f}.js`);
