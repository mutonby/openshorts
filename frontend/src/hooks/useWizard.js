// Generic wizard state machine. Holds { step, data } via useReducer with
// optional localStorage persistence keyed by `storageKey`. Step components
// stay dumb — they receive { step, data, setData, next, back, goto, reset }.
//
// Steps array:
//   [
//     { id: 'upload',     label: 'Upload' },
//     { id: 'categorize', label: 'Categorize' },
//     { id: 'processing', label: 'Processing', lock: true },
//     { id: 'review',     label: 'Review' },
//   ]
//
// `lock: true` disables BACK while on that step (used for Processing — you
// can't undo work in flight; only forward/skip or reset).

import { useEffect, useReducer, useRef } from 'react';

function reducer(state, action) {
  switch (action.type) {
    case 'NEXT':
      return { ...state, step: Math.min(state.step + 1, state.maxStep) };
    case 'BACK':
      return { ...state, step: Math.max(0, state.step - 1) };
    case 'GOTO':
      return { ...state, step: Math.max(0, Math.min(action.step, state.maxStep)) };
    case 'SET_DATA':
      return {
        ...state,
        data: typeof action.data === 'function'
          ? action.data(state.data)
          : { ...state.data, ...action.data },
      };
    case 'RESET':
      return { ...state, step: 0, data: action.initialData };
    case 'REHYDRATE':
      return action.state;
    default:
      return state;
  }
}

export function useWizard({ steps, initialData = {}, storageKey = null, resetOnRehydrate = null }) {
  const maxStep = steps.length - 1;

  const initial = useRef({
    step: 0,
    data: initialData,
    maxStep,
  });

  const [state, dispatch] = useReducer(reducer, initial.current);

  // Rehydrate once from localStorage. File objects don't survive JSON
  // round-trips. If `resetOnRehydrate(mergedData)` returns true, force
  // step=0 + initialData and clear persistence — keeps users from
  // marching past lost state into a step that will fail.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved && typeof saved.step === 'number') {
        const merged = { ...initialData, ...(saved.data || {}) };
        const corrupt =
          saved.step > 0
          && typeof resetOnRehydrate === 'function'
          && resetOnRehydrate(merged);
        dispatch({
          type: 'REHYDRATE',
          state: {
            step: corrupt ? 0 : Math.min(saved.step, maxStep),
            data: corrupt ? initialData : merged,
            maxStep,
          },
        });
        if (corrupt) localStorage.removeItem(storageKey);
      }
    } catch {/* ignore */}
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  // Persist on every change.
  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify({
        step: state.step,
        data: state.data,
      }));
    } catch {/* localStorage full or non-serializable — ignore */}
  }, [state.step, state.data, storageKey]);

  const currentStep = steps[state.step];
  const isLocked = !!currentStep?.lock;

  return {
    step: state.step,
    currentStep,
    steps,
    data: state.data,
    setData: (patch) => dispatch({ type: 'SET_DATA', data: patch }),
    next: () => dispatch({ type: 'NEXT' }),
    back: () => { if (!isLocked) dispatch({ type: 'BACK' }); },
    goto: (s) => dispatch({ type: 'GOTO', step: s }),
    reset: () => {
      dispatch({ type: 'RESET', initialData });
      if (storageKey) localStorage.removeItem(storageKey);
    },
    canBack: state.step > 0 && !isLocked,
    canNext: state.step < maxStep,
    isFirst: state.step === 0,
    isLast: state.step === maxStep,
    isLocked,
  };
}
