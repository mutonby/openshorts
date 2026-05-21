// Self-contained Snake on a 20x20 grid to fill processing wait time.
// Pauses automatically when the tab is hidden. Arrow keys / WASD to steer,
// space to pause.

import { useEffect, useRef, useState } from 'react';

const GRID = 20;
const TICK_MS = 110;

function randCell(occupied) {
  for (let i = 0; i < 200; i++) {
    const c = { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
    if (!occupied.some((p) => p.x === c.x && p.y === c.y)) return c;
  }
  return { x: 0, y: 0 };
}

const INITIAL_SNAKE = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }];
const INITIAL_DIR = { x: 1, y: 0 };
const INITIAL_FOOD = { x: 14, y: 10 };

export default function SnakeGame({ onScore }) {
  const [snake, setSnake] = useState(INITIAL_SNAKE);
  const [dir, setDir] = useState(INITIAL_DIR);
  const [food, setFood] = useState(INITIAL_FOOD);
  const [score, setScore] = useState(0);
  const [running, setRunning] = useState(true);
  const [over, setOver] = useState(false);

  // Stash latest dir for the keydown closure.
  const dirRef = useRef(dir);
  dirRef.current = dir;

  useEffect(() => {
    function onKey(e) {
      const k = e.key;
      const cur = dirRef.current;
      if ((k === 'ArrowUp' || k === 'w' || k === 'W') && cur.y !== 1) { setDir({ x: 0, y: -1 }); e.preventDefault(); }
      else if ((k === 'ArrowDown' || k === 's' || k === 'S') && cur.y !== -1) { setDir({ x: 0, y: 1 }); e.preventDefault(); }
      else if ((k === 'ArrowLeft' || k === 'a' || k === 'A') && cur.x !== 1) { setDir({ x: -1, y: 0 }); e.preventDefault(); }
      else if ((k === 'ArrowRight' || k === 'd' || k === 'D') && cur.x !== -1) { setDir({ x: 1, y: 0 }); e.preventDefault(); }
      else if (k === ' ') { setRunning((r) => !r); e.preventDefault(); }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    function onVis() { if (document.hidden) setRunning(false); }
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  useEffect(() => {
    if (!running || over) return;
    const id = setInterval(() => {
      setSnake((prev) => {
        const head = prev[0];
        const next = { x: head.x + dir.x, y: head.y + dir.y };
        if (next.x < 0 || next.x >= GRID || next.y < 0 || next.y >= GRID) {
          setOver(true); setRunning(false); return prev;
        }
        if (prev.some((p) => p.x === next.x && p.y === next.y)) {
          setOver(true); setRunning(false); return prev;
        }
        const ate = next.x === food.x && next.y === food.y;
        const newSnake = [next, ...prev];
        if (!ate) newSnake.pop();
        else {
          setScore((s) => { const v = s + 1; onScore?.(v); return v; });
          setFood(randCell(newSnake));
        }
        return newSnake;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [running, over, dir, food, onScore]);

  function reset() {
    setSnake(INITIAL_SNAKE);
    setDir(INITIAL_DIR);
    setFood(INITIAL_FOOD);
    setScore(0);
    setOver(false);
    setRunning(true);
  }

  return (
    <div className="select-none">
      <div className="flex items-center justify-between mb-2 text-[11px] text-zinc-400">
        <span>Score: <span className="text-white font-mono">{score}</span></span>
        <span className="text-zinc-600">
          {over ? 'Game over' : running ? 'playing' : 'paused — space to resume'}
        </span>
      </div>
      <div
        className="grid bg-black border border-border rounded-md overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)`, aspectRatio: '1 / 1' }}
      >
        {Array.from({ length: GRID * GRID }).map((_, i) => {
          const x = i % GRID;
          const y = Math.floor(i / GRID);
          const isHead = snake[0].x === x && snake[0].y === y;
          const isBody = !isHead && snake.some((p) => p.x === x && p.y === y);
          const isFood = food.x === x && food.y === y;
          return (
            <div
              key={i}
              className={
                isHead ? 'bg-primary' :
                isBody ? 'bg-primary/60' :
                isFood ? 'bg-success rounded-sm' :
                'bg-zinc-900/40'
              }
            />
          );
        })}
      </div>
      <div className="mt-3 flex items-center justify-between text-[11px] text-zinc-500">
        <span>Arrow keys / WASD · space to pause</span>
        {over && (
          <button onClick={reset} className="px-2 py-1 rounded-md bg-primary/20 text-primary text-[11px] hover:bg-primary/30">
            Play again
          </button>
        )}
      </div>
    </div>
  );
}
