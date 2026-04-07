const MAX_ROWS = 45;
const MAX_COLS = 45;
const WALL = 0;
const PATH = 1;

const ACTIONS = [
  { name: 'UP', dr: -1, dc: 0 },
  { name: 'DOWN', dr: 1, dc: 0 },
  { name: 'LEFT', dr: 0, dc: -1 },
  { name: 'RIGHT', dr: 0, dc: 1 }
];

const WATCH_SCHEDULE = [];
const DEFAULT_TRAIN_EPISODES_PER_RUN = 1000;
const REPLAY_MEMORY_SIZE = 10000;
const REPLAY_BATCH_SIZE = 64;
const REPLAY_INTERVAL = 4;
const SAVE_INTERVAL = 100;
const UI_UPDATE_INTERVAL = 20;
const EVAL_INTERVAL = 200;
const BACKGROUND_EVAL_EPISODES = 4;
const DEFAULT_EVAL_EPISODES = 12;
const RECENT_WINDOW = 25;
const TRAIN_VISUAL_STEP_DELAY = 24;
const WATCH_STEP_DELAY = 16;
const WATCH_STEPS_PER_TICK = 10;
const STORAGE_META_KEY = 'maze-hybrid-agent-meta-v2';
const STORAGE_LATEST_MODEL = 'indexeddb://maze-hybrid-latest';
const STORAGE_BEST_MODEL = 'indexeddb://maze-hybrid-best';
const LEGACY_STORAGE_LATEST_MODEL = 'localstorage://maze-hybrid-latest';
const LEGACY_STORAGE_BEST_MODEL = 'localstorage://maze-hybrid-best';
const TRAIN_SEED_BASE = 1337;
const EVAL_SEEDS = [50101, 50123, 50147, 50177, 50207, 50231, 50261, 50287, 50311, 50329, 50359, 50377];
const STATE_SIZE = 36;
const N_STEP_RETURNS = 3;
const PLANNER_GUIDANCE_WEIGHT = 0.18;

const CURRICULUM_STAGES = [
  {
    label: '1',
    rows: 25,
    cols: 25,
    monsterMoveEvery: 6,
    monsterWarmupSteps: 20,
    roomCount: 12,
    roomMin: 3,
    roomMax: 6,
    wallKnockRatio: 0.14,
    monsterSpawnThreshold: 0.78,
    stepMultiplier: 2.2
  },
  {
    label: '2',
    rows: 35,
    cols: 35,
    monsterMoveEvery: 4,
    monsterWarmupSteps: 14,
    roomCount: 12,
    roomMin: 3,
    roomMax: 5,
    wallKnockRatio: 0.11,
    monsterSpawnThreshold: 0.62,
    stepMultiplier: 1.95
  },
  {
    label: '3',
    rows: MAX_ROWS,
    cols: MAX_COLS,
    monsterMoveEvery: 3,
    monsterWarmupSteps: 10,
    roomCount: 10,
    roomMin: 3,
    roomMax: 5,
    wallKnockRatio: 0.08,
    monsterSpawnThreshold: 0.48,
    stepMultiplier: 1.75
  }
];

let grid = [];
let cells = [];
let reachablePathCount = 0;
let stepLimit = 0;
let activeRows = 0;
let activeCols = 0;

let pSpawn = { r: 1, c: 1 };
let mSpawn = { r: 1, c: 1 };
let exitPos = { r: 1, c: 1 };
let personPos = { ...pSpawn };
let monsterPos = { ...mSpawn };

let agent = null;
let metrics = null;
let currentEpisode = null;
let currentMazeStage = CURRICULUM_STAGES[0];
let isRunning = false;
let runMode = 'idle';
let loopTimer = null;
let visualEpisodeNumber = null;
let stopRequested = false;
let epsilon = 1;
let trainingWorker = null;
let workerReady = false;
let workerTrainingState = null;
const minEpsilon = 0.05;
const epsilonDecay = 0.995;

const mazeContainer = document.getElementById('maze');
const trainBtn = document.getElementById('train-btn');
const watchBtn = document.getElementById('watch-btn');
const evalBtn = document.getElementById('eval-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');
const trainEpisodesInput = document.getElementById('train-episodes-input');
const evalEpisodesInput = document.getElementById('eval-episodes-input');
const statusDisplay = document.getElementById('status');
const statsDisplay = document.getElementById('stats');
const trainingSummary = document.getElementById('training-summary');

const metricMazeSize = document.getElementById('metric-maze-size');
const metricMazeNote = document.getElementById('metric-maze-note');
const metricGamesPlayed = document.getElementById('metric-games-played');
const metricWinLoss = document.getElementById('metric-win-loss');
const metricWinLossNote = document.getElementById('metric-win-loss-note');
const metricLast25 = document.getElementById('metric-last-25');
const detailEpisodes = document.getElementById('detail-episodes');
const detailWins = document.getElementById('detail-wins');
const detailRecent = document.getElementById('detail-recent');
const detailEval = document.getElementById('detail-eval');
const detailStage = document.getElementById('detail-stage');
const detailSteps = document.getElementById('detail-steps');
const detailCoverage = document.getElementById('detail-coverage');
const detailModel = document.getElementById('detail-model');
const detailLastLoss = document.getElementById('detail-last-loss');
const detailMazeSize = document.getElementById('detail-maze-size');
const detailEvalSummary = document.getElementById('detail-eval-summary');
const detailDeaths = document.getElementById('detail-deaths');
const detailStalls = document.getElementById('detail-stalls');
const detailSource = document.getElementById('detail-source');
const detailRunType = document.getElementById('detail-run-type');
const detailCheckpoint = document.getElementById('detail-checkpoint');
const detailDiscoveries = document.getElementById('detail-discoveries');
const detailFrontier = document.getElementById('detail-frontier');
const detailRepeat = document.getElementById('detail-repeat');
const detailOscillation = document.getElementById('detail-oscillation');
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanels = document.querySelectorAll('.tab-panel');

class RNG {
  constructor(seed) {
    this.seed = seed >>> 0;
  }

  next() {
    let value = this.seed += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  }

  int(maxExclusive) {
    return Math.floor(this.next() * maxExclusive);
  }
}

class PrioritizedReplayBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = [];
    this.priorities = [];
    this.index = 0;
  }

  push(experience, priority) {
    const safePriority = Math.max(priority, 0.01);
    if (this.buffer.length < this.capacity) {
      this.buffer.push(experience);
      this.priorities.push(safePriority);
      return;
    }

    this.buffer[this.index] = experience;
    this.priorities[this.index] = safePriority;
    this.index = (this.index + 1) % this.capacity;
  }

  sample(batchSize) {
    const actualSize = Math.min(batchSize, this.buffer.length);
    const sample = [];
    const used = new Set();
    let totalPriority = this.priorities.reduce((sum, value) => sum + value, 0);

    while (sample.length < actualSize && used.size < this.buffer.length) {
      let target = Math.random() * totalPriority;
      for (let index = 0; index < this.buffer.length; index += 1) {
        if (used.has(index)) {
          continue;
        }
        target -= this.priorities[index];
        if (target <= 0) {
          used.add(index);
          sample.push({ index, experience: this.buffer[index] });
          totalPriority -= this.priorities[index];
          break;
        }
      }
    }

    return sample;
  }

  updatePriority(index, priority) {
    if (index >= 0 && index < this.priorities.length) {
      this.priorities[index] = Math.max(priority, 0.01);
    }
  }

  get size() {
    return this.buffer.length;
  }
}

class RewardNormalizer {
  constructor() {
    this.absMean = 1;
    this.sampleCount = 0;
  }

  observe(reward) {
    this.sampleCount += 1;
    const smoothing = this.sampleCount < 50 ? 0.08 : 0.02;
    this.absMean += (Math.abs(reward) - this.absMean) * smoothing;
  }

  normalize(reward) {
    const scale = Math.max(1, this.absMean * 2.5);
    return Math.max(-4, Math.min(4, reward / scale));
  }

  snapshot() {
    return {
      absMean: this.absMean,
      sampleCount: this.sampleCount
    };
  }

  restore(snapshot) {
    if (!snapshot) {
      return;
    }
    this.absMean = Math.max(1, snapshot.absMean || 1);
    this.sampleCount = Math.max(0, snapshot.sampleCount || 0);
  }
}

class DQNAgent {
  constructor(stateSize, actionSize) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.gamma = 0.95;
    this.learningRate = 0.001;
    this.nStep = N_STEP_RETURNS;
    this.plannerGuidanceWeight = PLANNER_GUIDANCE_WEIGHT;
    this.memory = new PrioritizedReplayBuffer(REPLAY_MEMORY_SIZE);
    this.rewardNormalizer = new RewardNormalizer();
    this.nStepQueue = [];
    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.bestModel = this.createModel();
    this.syncTargetModel();
    this.syncBestModel();
    this.trainingSteps = 0;
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [this.stateSize], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(this.learningRate), loss: tf.losses.huberLoss });
    return model;
  }

  act(stateArray, epsilonValue, sourceModel = 'latest') {
    if (sourceModel === 'latest' && Math.random() < epsilonValue) {
      return Math.floor(Math.random() * this.actionSize);
    }

    const model = sourceModel === 'best' ? this.bestModel : this.model;
    return tf.tidy(() => {
      const input = tf.tensor2d([stateArray], [1, this.stateSize]);
      const output = model.predict(input);
      return output.argMax(1).dataSync()[0];
    });
  }

  startEpisode() {
    this.nStepQueue = [];
  }

  rememberTransition(state, action, reward, nextState, done, plannerAction = null, plannerStrength = 0) {
    this.rewardNormalizer.observe(reward);
    this.nStepQueue.push({
      state,
      action,
      reward: this.rewardNormalizer.normalize(reward),
      nextState,
      done,
      plannerAction,
      plannerStrength
    });

    if (this.nStepQueue.length >= this.nStep || done) {
      this.flushNStep(done);
    }

    if (done) {
      while (this.nStepQueue.length > 0) {
        this.flushNStep(true);
      }
    }
  }

  flushNStep(forceTail) {
    if (this.nStepQueue.length === 0) {
      return;
    }

    const span = forceTail ? this.nStepQueue.length : Math.min(this.nStep, this.nStepQueue.length);
    const first = this.nStepQueue[0];
    const last = this.nStepQueue[span - 1];
    let cumulativeReward = 0;

    for (let index = 0; index < span; index += 1) {
      cumulativeReward += this.nStepQueue[index].reward * (this.gamma ** index);
      if (this.nStepQueue[index].done) {
        break;
      }
    }

    const priority = Math.abs(cumulativeReward) + (last.done ? 2 : 0.3) + first.plannerStrength * 0.5;
    this.memory.push({
      state: first.state,
      action: first.action,
      reward: cumulativeReward,
      nextState: last.nextState,
      done: last.done,
      plannerAction: first.plannerAction,
      plannerStrength: first.plannerStrength,
      horizon: span
    }, priority);

    this.nStepQueue.shift();
  }

  async replay(batchSize) {
    if (this.memory.size < batchSize) {
      return;
    }

    const batch = this.memory.sample(batchSize);
    const states = batch.map((item) => item.experience.state);
    const nextStates = batch.map((item) => item.experience.nextState);

    const statesTensor = tf.tensor2d(states, [batch.length, this.stateSize]);
    const nextStatesTensor = tf.tensor2d(nextStates, [batch.length, this.stateSize]);
    const currentQTensor = this.model.predict(statesTensor);
    const onlineNextQTensor = this.model.predict(nextStatesTensor);
    const targetNextQTensor = this.targetModel.predict(nextStatesTensor);

    const currentQ = await currentQTensor.array();
    const onlineNextQ = await onlineNextQTensor.array();
    const targetNextQ = await targetNextQTensor.array();

    currentQTensor.dispose();
    onlineNextQTensor.dispose();
    targetNextQTensor.dispose();
    statesTensor.dispose();
    nextStatesTensor.dispose();

    for (let index = 0; index < batch.length; index += 1) {
      const sample = batch[index];
      const experience = sample.experience;
      let target = experience.reward;
      if (!experience.done) {
        const bestNextAction = argMax(onlineNextQ[index]);
        target += (this.gamma ** (experience.horizon || 1)) * targetNextQ[index][bestNextAction];
      }

      const tdError = Math.abs(target - currentQ[index][experience.action]);
      currentQ[index][experience.action] = target;

       if (experience.plannerAction !== null && experience.plannerStrength > 0) {
        const plannerBoost = Math.max(target, Math.max(...currentQ[index]) * 0.92 + experience.plannerStrength);
        currentQ[index][experience.plannerAction] =
          currentQ[index][experience.plannerAction] * (1 - this.plannerGuidanceWeight)
          + plannerBoost * this.plannerGuidanceWeight;
      }

      this.memory.updatePriority(sample.index, tdError + 0.01);
    }

    const fitStates = tf.tensor2d(states, [batch.length, this.stateSize]);
    const fitTargets = tf.tensor2d(currentQ, [batch.length, this.actionSize]);
    await this.model.fit(fitStates, fitTargets, {
      batchSize: batch.length,
      epochs: 1,
      shuffle: true,
      verbose: 0
    });
    fitStates.dispose();
    fitTargets.dispose();

    this.trainingSteps += 1;
    if (this.trainingSteps % 10 === 0) {
      this.syncTargetModel();
    }
  }

  syncTargetModel() {
    const weights = this.model.getWeights().map((weight) => weight.clone());
    this.targetModel.setWeights(weights);
    weights.forEach((weight) => weight.dispose());
  }

  syncBestModel() {
    const weights = this.model.getWeights().map((weight) => weight.clone());
    this.bestModel.setWeights(weights);
    weights.forEach((weight) => weight.dispose());
  }

  async saveLatest() {
    await this.model.save(STORAGE_LATEST_MODEL);
  }

  async saveBest() {
    await this.bestModel.save(STORAGE_BEST_MODEL);
  }

  async loadLatest() {
    try {
      this.model = await tf.loadLayersModel(STORAGE_LATEST_MODEL);
      this.model.compile({ optimizer: tf.train.adam(this.learningRate), loss: tf.losses.huberLoss });
      this.targetModel = this.createModel();
      this.bestModel = this.createModel();
      this.syncTargetModel();
      return true;
    } catch {
      return false;
    }
  }

  async loadBest() {
    try {
      this.bestModel = await tf.loadLayersModel(STORAGE_BEST_MODEL);
      this.bestModel.compile({ optimizer: tf.train.adam(this.learningRate), loss: tf.losses.huberLoss });
      return true;
    } catch {
      this.syncBestModel();
      return false;
    }
  }

  async dispose() {
    this.model.dispose();
    this.targetModel.dispose();
    this.bestModel.dispose();
  }

  getMeta() {
    return {
      rewardNormalizer: this.rewardNormalizer.snapshot(),
      trainingSteps: this.trainingSteps
    };
  }

  restoreMeta(meta) {
    if (!meta) {
      return;
    }
    this.rewardNormalizer.restore(meta.rewardNormalizer);
    this.trainingSteps = meta.trainingSteps || 0;
  }
}

class WorldMemory {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.occupancy = Array.from({ length: rows }, () => Array(cols).fill('unknown'));
    this.explored = new Set();
    this.frontiers = new Set();
    this.safeFrontiers = new Set();
    this.riskyFrontiers = new Set();
    this.revisits = new Map();
    this.dangerMap = Array.from({ length: rows }, () => Array(cols).fill(Number.POSITIVE_INFINITY));
    this.exitSeen = false;
    this.exitPosition = null;
    this.pathHistory = [];
    this.noProgressSteps = 0;
    this.lastNoveltyStep = 0;
    this.lastSafetyStep = 0;
  }

  observe(position, gridRef, exitRef, monsterRef, stepCount) {
    let newDiscoveries = 0;
    const offsets = [
      { dr: 0, dc: 0 },
      { dr: -1, dc: 0 },
      { dr: 1, dc: 0 },
      { dr: 0, dc: -1 },
      { dr: 0, dc: 1 }
    ];

    for (const offset of offsets) {
      const row = position.r + offset.dr;
      const col = position.c + offset.dc;
      if (!isInside(row, col)) {
        continue;
      }
      const key = `${row}_${col}`;
      const discoveredType = gridRef[row][col] === PATH ? 'path' : 'wall';
      if (this.occupancy[row][col] === 'unknown' && discoveredType === 'path') {
        newDiscoveries += 1;
      }
      this.occupancy[row][col] = discoveredType;
      if (discoveredType === 'path') {
        this.explored.add(key);
      }
      if (row === exitRef.r && col === exitRef.c) {
        this.exitSeen = true;
        this.exitPosition = { ...exitRef };
      }
    }

    const posKey = keyFor(position);
    this.revisits.set(posKey, (this.revisits.get(posKey) || 0) + 1);
    this.pathHistory.push(posKey);
    if (this.pathHistory.length > 16) {
      this.pathHistory.shift();
    }

    if (newDiscoveries > 0) {
      this.lastNoveltyStep = stepCount;
      this.noProgressSteps = 0;
    } else {
      this.noProgressSteps += 1;
    }

    this.updateDanger(monsterRef, gridRef, stepCount);
    this.rebuildFrontiers(position);
    return newDiscoveries;
  }

  updateDanger(monsterRef, gridRef, stepCount) {
    const distances = bfsDistances(monsterRef, gridRef, this.rows, this.cols, true);
    this.dangerMap = distances;
    const currentDanger = distances[personPos.r][personPos.c];
    if (Number.isFinite(currentDanger)) {
      this.lastSafetyStep = stepCount;
    }
  }

  rebuildFrontiers(currentPosition) {
    this.frontiers.clear();
    this.safeFrontiers.clear();
    this.riskyFrontiers.clear();

    for (const key of this.explored) {
      const [rowText, colText] = key.split('_');
      const row = Number(rowText);
      const col = Number(colText);
      let isFrontier = false;
      for (const action of ACTIONS) {
        const nextRow = row + action.dr;
        const nextCol = col + action.dc;
        if (isInside(nextRow, nextCol) && this.occupancy[nextRow][nextCol] === 'unknown') {
          isFrontier = true;
          break;
        }
      }

      if (!isFrontier) {
        continue;
      }

      this.frontiers.add(key);
      const danger = this.dangerMap[row][col];
      const currentDanger = this.dangerMap[currentPosition.r][currentPosition.c];
      if (danger >= currentDanger + 2 || !Number.isFinite(currentDanger)) {
        this.safeFrontiers.add(key);
      } else {
        this.riskyFrontiers.add(key);
      }
    }
  }

  hasSafePlannedPath(position) {
    if (this.exitSeen && this.exitPosition) {
      const path = Planner.findPath(position, (node) => positionsMatch(node, this.exitPosition), this, 'exit');
      if (path.length > 0) {
        return true;
      }
    }

    return this.getBestSafeFrontierPath(position).length > 0;
  }

  getBestSafeFrontierPath(position) {
    const candidateSet = this.safeFrontiers.size > 0 ? this.safeFrontiers : this.frontiers;
    let bestPath = [];
    let bestScore = Number.POSITIVE_INFINITY;

    for (const key of candidateSet) {
      const [rowText, colText] = key.split('_');
      const target = { r: Number(rowText), c: Number(colText) };
      const path = Planner.findPath(position, (node) => positionsMatch(node, target), this, 'frontier');
      if (path.length === 0) {
        continue;
      }

      const score = Planner.scorePath(path, this, 'frontier');
      if (score < bestScore) {
        bestScore = score;
        bestPath = path;
      }
    }

    return bestPath;
  }

  detectLoopOrStall(stepCount) {
    if (this.noProgressSteps >= 12) {
      return true;
    }

    if (this.pathHistory.length < 8) {
      return false;
    }

    const tail = this.pathHistory.slice(-8);
    const patternA = tail.slice(0, 4).join('|');
    const patternB = tail.slice(4).join('|');
    if (patternA === patternB && stepCount - this.lastNoveltyStep > 4) {
      return true;
    }

    return false;
  }
}

class Planner {
  static findPath(start, goalTest, memory, mode) {
    const startKey = keyFor(start);
    const open = [{ node: { ...start }, key: startKey, g: 0, f: 0 }];
    const cameFrom = new Map();
    const gScore = new Map([[startKey, 0]]);
    const visited = new Set();

    while (open.length > 0) {
      open.sort((left, right) => left.f - right.f);
      const current = open.shift();
      if (!current) {
        break;
      }

      if (goalTest(current.node)) {
        return Planner.reconstructPath(cameFrom, current.node);
      }

      if (visited.has(current.key)) {
        continue;
      }
      visited.add(current.key);

      for (const action of ACTIONS) {
        const next = { r: current.node.r + action.dr, c: current.node.c + action.dc };
        const nextKey = keyFor(next);
        if (!Planner.isTraversable(next, memory)) {
          continue;
        }

        const tentativeG = current.g + Planner.stepCost(next, memory, mode);
        if (tentativeG >= (gScore.get(nextKey) ?? Number.POSITIVE_INFINITY)) {
          continue;
        }

        cameFrom.set(nextKey, current.node);
        gScore.set(nextKey, tentativeG);
        open.push({ node: next, key: nextKey, g: tentativeG, f: tentativeG + Planner.heuristic(next, memory, mode) });
      }
    }

    return [];
  }

  static isTraversable(node, memory) {
    return isInside(node.r, node.c) && memory.occupancy[node.r][node.c] === 'path';
  }

  static stepCost(node, memory, mode) {
    const danger = memory.dangerMap[node.r][node.c];
    const revisitPressure = memory.revisits.get(keyFor(node)) || 0;
    let cost = 1 + revisitPressure * 0.35;
    if (Number.isFinite(danger)) {
      cost += Math.max(0, 9 - danger) * 1.25;
    }
    if (mode === 'frontier' && memory.safeFrontiers.has(keyFor(node))) {
      cost -= 0.35;
    }
    return cost;
  }

  static heuristic(node, memory, mode) {
    if (mode === 'exit' && memory.exitPosition) {
      return manhattan(node, memory.exitPosition);
    }
    return 0;
  }

  static reconstructPath(cameFrom, current) {
    const path = [{ ...current }];
    let cursor = current;
    while (cameFrom.has(keyFor(cursor))) {
      cursor = cameFrom.get(keyFor(cursor));
      path.unshift({ ...cursor });
    }
    return path;
  }

  static scorePath(path, memory, mode) {
    let score = 0;
    for (let index = 1; index < path.length; index += 1) {
      score += Planner.stepCost(path[index], memory, mode);
    }
    return score;
  }
}

function createMetrics() {
  return {
    gamesPlayed: 0,
    gamesWon: 0,
    gameOutcomes: [],
    totalEpisodes: 0,
    trainingWins: 0,
    heldOutEvalWinRate: 0,
    bestEvalWinRate: 0,
    bestCheckpointEpisode: 0,
    bestCheckpointImproved: false,
    totalCoverage: 0,
    totalWinSteps: 0,
    winEpisodes: 0,
    recentOutcomes: [],
    curriculumStageIndex: 0,
    monsterDeaths: 0,
    stallFailures: 0,
    oscillationFailures: 0,
    lastEvalSummary: 'none',
    frontierDiscoveries: 0,
    safeFrontierPicks: 0,
    loopRecoveries: 0,
    lastLossReason: 'none',
    lastActionSource: 'policy',
    lastRepeatStreak: 0,
    lastOscillationCount: 0,
    lastRunType: 'idle',
    configuredTrainEpisodes: DEFAULT_TRAIN_EPISODES_PER_RUN,
    configuredEvalEpisodes: DEFAULT_EVAL_EPISODES
  };
}

function createEpisodeState() {
  return {
    stepCount: 0,
    visitedCounts: new Map(),
    heatVisits: new Map(),
    lastAction: null,
    repeatActionStreak: 0,
    oscillationCount: 0,
    actionSource: 'policy',
    world: new WorldMemory(activeRows, activeCols)
  };
}

metrics = createMetrics();
currentEpisode = createEpisodeState();

async function init() {
  if (!window.tf) {
    statusDisplay.textContent = 'TensorFlow.js failed to load.';
    statusDisplay.style.color = 'var(--monster-color)';
    return;
  }

  trainBtn.addEventListener('click', startTrainingCycle);
  watchBtn.addEventListener('click', () => {
    void startWatchCycle();
  });
  evalBtn.addEventListener('click', () => {
    void startEvaluationCycle();
  });
  stopBtn.addEventListener('click', requestStop);
  resetBtn.addEventListener('click', hardReset);
  trainEpisodesInput.addEventListener('change', syncConfiguredRunsFromInputs);
  evalEpisodesInput.addEventListener('change', syncConfiguredRunsFromInputs);
  tabButtons.forEach((button) => {
    button.addEventListener('click', () => {
      setActiveTab(button.dataset.tabTarget);
    });
  });

  agent = new DQNAgent(STATE_SIZE, ACTIONS.length);
  await loadProgress();
  await initializeTrainingWorker();
  syncInputsFromMetrics();

  if (metrics.totalEpisodes > 0) {
    statusDisplay.textContent = 'Loaded saved hybrid-agent progress. Continue training or watch the best checkpoint.';
  } else {
    statusDisplay.textContent = 'Ready for hybrid-agent training across unknown mazes.';
  }
  statusDisplay.style.color = 'var(--text-color)';

  currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
  generateMaze(currentMazeStage, TRAIN_SEED_BASE);
  syncMazeDom();
  resetEpisode();
  runSelfChecks();
  drawGrid();
  updateStats('idle');
}

async function hardReset() {
  stopSimulation();
  if (trainingWorker) {
    trainingWorker.terminate();
    trainingWorker = null;
    workerReady = false;
    workerTrainingState = null;
  }
  if (agent) {
    await agent.dispose();
  }
  await clearProgress();
  agent = new DQNAgent(STATE_SIZE, ACTIONS.length);
  metrics = createMetrics();
  epsilon = 1;
  currentMazeStage = CURRICULUM_STAGES[0];
  syncInputsFromMetrics();
  generateMaze(currentMazeStage, TRAIN_SEED_BASE);
  syncMazeDom();
  resetEpisode();
  drawGrid();
  await initializeTrainingWorker();
  statusDisplay.style.color = 'var(--text-color)';
  statusDisplay.textContent = 'Training state cleared. Ready for a new experiment.';
  updateStats('idle');
}

function startTrainingCycle() {
  if (isRunning || !agent) {
    return;
  }

  syncConfiguredRunsFromInputs();
  isRunning = true;
  stopRequested = false;
  runMode = 'train';
  workerTrainingState = null;
  metrics.lastRunType = 'training';
  metrics.bestCheckpointImproved = false;
  trainBtn.disabled = true;
  watchBtn.disabled = true;
  evalBtn.disabled = true;
  stopBtn.disabled = false;
  statusDisplay.style.color = 'var(--text-color)';
  statusDisplay.textContent = `Training hybrid agent across fresh mazes for ${metrics.configuredTrainEpisodes} episodes.`;
  updateStats('train');

  if (!trainingWorker || !workerReady) {
    runTrainingLoop().catch((error) => {
      console.error(error);
      statusDisplay.style.color = 'var(--monster-color)';
      statusDisplay.textContent = 'Training failed. Check the console for details.';
      stopSimulation();
    });
    return;
  }

  trainingWorker.postMessage({
    type: 'start-training',
    payload: {
      trainEpisodes: metrics.configuredTrainEpisodes,
      evalEpisodes: metrics.configuredEvalEpisodes
    }
  });
}

async function runTrainingLoop() {
  const targetEpisode = metrics.totalEpisodes + metrics.configuredTrainEpisodes;

  while (isRunning && !stopRequested && metrics.totalEpisodes < targetEpisode) {
    maybeAdvanceCurriculum();
    currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
    const episodeNumber = metrics.totalEpisodes + 1;
    const seed = TRAIN_SEED_BASE + episodeNumber * 7919;
    generateMaze(currentMazeStage, seed);
    resetEpisode();

    if (WATCH_SCHEDULE.includes(episodeNumber)) {
      visualEpisodeNumber = episodeNumber;
      syncMazeDom();
      statusDisplay.textContent = `Watching training episode ${episodeNumber}.`;
      await runVisualTrainingEpisode();
    } else {
      visualEpisodeNumber = null;
      runHeadlessTrainingEpisode();
    }

    if (metrics.totalEpisodes % REPLAY_INTERVAL === 0) {
      await agent.replay(REPLAY_BATCH_SIZE);
    }
    epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);

    if (metrics.totalEpisodes % EVAL_INTERVAL === 0) {
      await runHeldOutEvaluation(BACKGROUND_EVAL_EPISODES);
      await saveProgress();
    } else if (metrics.totalEpisodes % SAVE_INTERVAL === 0) {
      await saveProgress();
    }

    if (metrics.totalEpisodes % UI_UPDATE_INTERVAL === 0 || visualEpisodeNumber !== null) {
      updateStats('train');
    }

    if (metrics.totalEpisodes % UI_UPDATE_INTERVAL === 0) {
      await tf.nextFrame();
    }
  }

  await saveProgress();
  statusDisplay.style.color = stopRequested ? 'var(--text-color)' : 'var(--exit-color)';
  statusDisplay.textContent = stopRequested
    ? 'Training stopped. The latest checkpoint and metrics were saved.'
    : 'Training run finished. The best checkpoint is ready for unseen-maze evaluation.';
  stopSimulation();
}

function runHeadlessTrainingEpisode() {
  agent.startEpisode();
  while (true) {
    if (!isRunning || stopRequested) {
      break;
    }
    const state = encodeStateFeatures();
    const decision = chooseHybridAction(false, 'latest');
    const outcome = environmentStep(decision.action, decision.source, false);
    const nextState = encodeStateFeatures();
    agent.rememberTransition(state, decision.action, outcome.reward, nextState, outcome.done, decision.plannerAction, decision.plannerStrength);
    if (outcome.done) {
      finishTrainingEpisode(outcome);
      break;
    }
  }
}

function runVisualTrainingEpisode() {
  agent.startEpisode();
  return new Promise((resolve) => {
    const step = () => {
      if (!isRunning || runMode !== 'train' || stopRequested) {
        resolve();
        return;
      }

      const state = encodeStateFeatures();
      const decision = chooseHybridAction(false, 'latest');
      const outcome = environmentStep(decision.action, decision.source, true);
      const nextState = encodeStateFeatures();
      agent.rememberTransition(state, decision.action, outcome.reward, nextState, outcome.done, decision.plannerAction, decision.plannerStrength);

      drawGrid();
      updateStats('train');

      if (outcome.done) {
        finishTrainingEpisode(outcome);
        resolve();
        return;
      }

      loopTimer = window.setTimeout(step, TRAIN_VISUAL_STEP_DELAY);
    };

    loopTimer = window.setTimeout(step, TRAIN_VISUAL_STEP_DELAY);
  });
}

async function runHeldOutEvaluation(episodeCount = metrics.configuredEvalEpisodes, resumeTraining = true) {
  let wins = 0;
  let completedEpisodes = 0;
  const reasonCounts = {
    escaped: 0,
    caught: 0,
    stall: 0,
    oscillation: 0,
    timeout: 0
  };

  for (let index = 0; index < episodeCount; index += 1) {
    if (stopRequested) {
      break;
    }
    generateMaze(CURRICULUM_STAGES[CURRICULUM_STAGES.length - 1], EVAL_SEEDS[index % EVAL_SEEDS.length]);
    resetEpisode();
    metrics.lastRunType = 'evaluation';
    completedEpisodes += 1;
    statusDisplay.textContent = `Evaluating maze ${completedEpisodes}/${episodeCount} on held-out layouts.`;
    updateStats('eval');
    await tf.nextFrame();

    while (true) {
      if (stopRequested) {
        break;
      }
      const decision = chooseHybridAction(true, 'latest');
      const outcome = environmentStep(decision.action, decision.source, false);
      if (outcome.done) {
        recordGameOutcome(outcome.reason);
        reasonCounts[outcome.reason] = (reasonCounts[outcome.reason] || 0) + 1;
        if (outcome.reason === 'escaped') {
          wins += 1;
        }
        break;
      }
    }
  }

  metrics.heldOutEvalWinRate = completedEpisodes === 0 ? 0 : (wins / completedEpisodes) * 100;
  metrics.lastEvalSummary = formatEvalSummary(reasonCounts, completedEpisodes);
  metrics.bestCheckpointImproved = false;
  if (!stopRequested && metrics.heldOutEvalWinRate >= metrics.bestEvalWinRate) {
    metrics.bestEvalWinRate = metrics.heldOutEvalWinRate;
    metrics.bestCheckpointEpisode = metrics.totalEpisodes;
    metrics.bestCheckpointImproved = true;
    agent.syncBestModel();
    await agent.saveBest();
  }

  if (resumeTraining) {
    currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
    generateMaze(currentMazeStage, TRAIN_SEED_BASE + metrics.totalEpisodes * 7919 + 3);
    resetEpisode();
    metrics.lastRunType = 'training';
  }
}

async function startEvaluationCycle() {
  if (isRunning || !agent) {
    return;
  }

  workerTrainingState = null;
  await refreshMainAgentModels();
  syncConfiguredRunsFromInputs();
  isRunning = true;
  stopRequested = false;
  runMode = 'eval';
  metrics.lastRunType = 'evaluation';
  trainBtn.disabled = true;
  watchBtn.disabled = true;
  evalBtn.disabled = true;
  stopBtn.disabled = false;
  statusDisplay.style.color = 'var(--text-color)';
  statusDisplay.textContent = `Running held-out evaluation on ${metrics.configuredEvalEpisodes} mazes.`;

  runExplicitEvaluation().catch((error) => {
    console.error(error);
    statusDisplay.style.color = 'var(--monster-color)';
    statusDisplay.textContent = 'Evaluation failed. Check the console for details.';
    stopSimulation();
  });
}

async function runExplicitEvaluation() {
  await runHeldOutEvaluation(metrics.configuredEvalEpisodes, false);
  await saveProgress();

  if (stopRequested) {
    statusDisplay.style.color = 'var(--text-color)';
    statusDisplay.textContent = 'Evaluation stopped.';
  } else {
    statusDisplay.style.color = 'var(--exit-color)';
    statusDisplay.textContent = `Held-out evaluation finished at ${metrics.heldOutEvalWinRate.toFixed(1)}% win rate.`;
  }

  stopSimulation();
}

async function startWatchCycle() {
  if (isRunning || !agent) {
    return;
  }

  workerTrainingState = null;
  await refreshMainAgentModels();
  isRunning = true;
  stopRequested = false;
  runMode = 'watch';
  metrics.lastRunType = 'watch';
  currentMazeStage = CURRICULUM_STAGES[CURRICULUM_STAGES.length - 1];
  generateMaze(currentMazeStage, EVAL_SEEDS[0]);
  syncMazeDom();
  resetEpisode();
  trainBtn.disabled = true;
  watchBtn.disabled = true;
  evalBtn.disabled = true;
  stopBtn.disabled = false;
  statusDisplay.style.color = 'var(--text-color)';
  statusDisplay.textContent = 'Watching the best checkpoint on a held-out maze.';
  drawGrid();
  updateStats('watch');

  const step = () => {
    if (!isRunning || runMode !== 'watch' || stopRequested) {
      if (stopRequested) {
        stopSimulation();
      }
      return;
    }

    let outcome = null;
    for (let index = 0; index < WATCH_STEPS_PER_TICK; index += 1) {
      const decision = chooseHybridAction(true, 'best');
      outcome = environmentStep(decision.action, decision.source, false);
      if (outcome.done) {
        break;
      }
    }

    drawGrid();
    updateStats('watch');

    if (outcome && outcome.done) {
      finishWatch(outcome);
      return;
    }

    loopTimer = window.setTimeout(step, WATCH_STEP_DELAY);
  };

  loopTimer = window.setTimeout(step, WATCH_STEP_DELAY);
}

function requestStop() {
  if (!isRunning) {
    return;
  }

  stopRequested = true;
  statusDisplay.style.color = 'var(--text-color)';
  statusDisplay.textContent = 'Stop requested. Ending the current run cleanly.';

  if (runMode === 'train' && trainingWorker) {
    trainingWorker.postMessage({ type: 'stop' });
  }
}

function finishWatch(outcome) {
  recordGameOutcome(outcome.reason);
  const labels = {
    escaped: 'Best checkpoint escaped the held-out maze.',
    caught: 'Best checkpoint was eaten by the monster.',
    stall: 'Best checkpoint stalled without useful progress.',
    oscillation: 'Best checkpoint fell into an oscillation pattern.',
    timeout: 'Best checkpoint timed out on the held-out maze.'
  };
  statusDisplay.style.color = outcome.reason === 'escaped' ? 'var(--exit-color)' : 'var(--monster-color)';
  statusDisplay.textContent = labels[outcome.reason] || 'Watch run ended.';
  stopSimulation();
}

function stopSimulation() {
  isRunning = false;
  runMode = 'idle';
  visualEpisodeNumber = null;
  stopRequested = false;
  workerTrainingState = null;
  if (loopTimer !== null) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
  trainBtn.disabled = false;
  watchBtn.disabled = false;
  evalBtn.disabled = false;
  stopBtn.disabled = true;
  updateStats('idle');
}

function chooseHybridAction(greedyOnly, sourceModel) {
  const world = currentEpisode.world;
  const loopRecovery = world.detectLoopOrStall(currentEpisode.stepCount);

  if (loopRecovery) {
    const recovery = choosePlannerAction('loop-recovery');
    if (recovery !== null) {
      metrics.loopRecoveries += 1;
      return { action: recovery, source: 'loop-recovery', plannerAction: recovery, plannerStrength: 1 };
    }
  }

  if (world.exitSeen && world.exitPosition) {
    const exitAction = choosePlannerAction('exit');
    if (exitAction !== null) {
      return { action: exitAction, source: 'planner', plannerAction: exitAction, plannerStrength: 1 };
    }
  }

  const safeFrontierAction = choosePlannerAction('frontier');
  const frontierPlannerStrength = world.safeFrontiers.size > 0 ? 0.75 : 0.35;
  if (safeFrontierAction !== null && world.safeFrontiers.size > 0 && Math.random() < 0.45) {
    metrics.safeFrontierPicks += 1;
    return {
      action: safeFrontierAction,
      source: 'planner',
      plannerAction: safeFrontierAction,
      plannerStrength: frontierPlannerStrength
    };
  }

  const state = encodeStateFeatures();
  return {
    action: agent.act(state, greedyOnly ? 0 : epsilon, sourceModel),
    source: 'policy',
    plannerAction: safeFrontierAction,
    plannerStrength: safeFrontierAction === null ? 0 : frontierPlannerStrength
  };
}

function choosePlannerAction(mode) {
  let path = [];
  if (mode === 'exit' && currentEpisode.world.exitSeen && currentEpisode.world.exitPosition) {
    path = Planner.findPath(personPos, (node) => positionsMatch(node, currentEpisode.world.exitPosition), currentEpisode.world, 'exit');
  } else {
    path = currentEpisode.world.getBestSafeFrontierPath(personPos);
  }

  if (path.length < 2) {
    return null;
  }

  const next = path[1];
  return actionIndexForMove(personPos, next);
}

function environmentStep(actionIndex, source, render) {
  currentEpisode.stepCount += 1;
  currentEpisode.actionSource = source;
  metrics.lastActionSource = source;

  let reward = 0;
  let done = false;
  let reason = 'running';

  const action = ACTIONS[actionIndex];
  const next = { r: personPos.r + action.dr, c: personPos.c + action.dc };
  const validMove = isPath(next.r, next.c);
  const previousDanger = currentEpisode.world.dangerMap[personPos.r][personPos.c];
  const previousDiscoveries = metrics.frontierDiscoveries;

  if (currentEpisode.lastAction === actionIndex) {
    currentEpisode.repeatActionStreak += 1;
  } else {
    currentEpisode.repeatActionStreak = 1;
  }

  if (isOppositeAction(currentEpisode.lastAction, actionIndex)) {
    currentEpisode.oscillationCount += 1;
  }

  if (!validMove) {
    reward -= 1.2;
  } else {
    personPos = next;
    rememberPathVisit(personPos);

    const key = keyFor(personPos);
    const visitCount = (currentEpisode.visitedCounts.get(key) || 0) + 1;
    currentEpisode.visitedCounts.set(key, visitCount);

    const discoveries = currentEpisode.world.observe(personPos, grid, exitPos, monsterPos, currentEpisode.stepCount);
    metrics.frontierDiscoveries += discoveries;
    if (discoveries > 0) {
      reward += discoveries * 4.5;
    }

    if (visitCount > 1) {
      reward -= 0.45;
    }

    if (isDeadEnd(personPos) && !positionsMatch(personPos, exitPos)) {
      reward -= 1.0;
    }

    if (positionsMatch(personPos, exitPos)) {
      reward += 100;
      done = true;
      reason = 'escaped';
    }
  }

  if (!done && shouldMonsterMove()) {
    moveMonsterGlobal();
    currentEpisode.world.updateDanger(monsterPos, grid, currentEpisode.stepCount);
    if (positionsMatch(personPos, monsterPos)) {
      reward -= 100;
      done = true;
      reason = 'caught';
    }
  }

  if (!done) {
    const currentDanger = currentEpisode.world.dangerMap[personPos.r][personPos.c];
    if (Number.isFinite(previousDanger) && Number.isFinite(currentDanger) && currentDanger > previousDanger) {
      reward += 0.8;
    }
  }

  if (!done && currentEpisode.world.detectLoopOrStall(currentEpisode.stepCount)) {
    done = true;
    reason = currentEpisode.oscillationCount >= 6 ? 'oscillation' : 'stall';
  }

  if (!done && currentEpisode.stepCount >= stepLimit) {
    done = true;
    reason = 'timeout';
  }

  currentEpisode.lastAction = actionIndex;
  metrics.lastRepeatStreak = currentEpisode.repeatActionStreak;
  metrics.lastOscillationCount = currentEpisode.oscillationCount;

  if (done) {
    if (reason === 'caught') {
      metrics.monsterDeaths += 1;
    } else if (reason === 'stall') {
      metrics.stallFailures += 1;
    } else if (reason === 'oscillation') {
      metrics.oscillationFailures += 1;
    }
    metrics.lastLossReason = reason === 'escaped' ? metrics.lastLossReason : reason;
  }

  if (render) {
    drawGrid();
  }

  return { reward, done, reason, newDiscoveries: metrics.frontierDiscoveries - previousDiscoveries };
}

function encodeStateFeatures() {
  const world = currentEpisode.world;
  const wallBits = ACTIONS.map((action) => (isPath(personPos.r + action.dr, personPos.c + action.dc) ? 1 : 0));
  const exploredBits = ACTIONS.map((action) => {
    const row = personPos.r + action.dr;
    const col = personPos.c + action.dc;
    return isInside(row, col) && world.occupancy[row][col] === 'path' ? 1 : 0;
  });
  const unknownBits = ACTIONS.map((action) => {
    const row = personPos.r + action.dr;
    const col = personPos.c + action.dc;
    return isInside(row, col) && world.occupancy[row][col] === 'unknown' ? 1 : 0;
  });

  const openCount = wallBits.reduce((sum, value) => sum + value, 0);
  const junctionType = getJunctionType(openCount);
  const frontierInfo = getFrontierInfo();
  const exitInfo = getExitInfo();
  const monsterInfo = getMonsterInfo();
  const revisitPressure = Math.min(currentEpisode.world.revisits.get(keyFor(personPos)) || 0, 6) / 6;
  const novelty = Math.max(0, 1 - currentEpisode.world.noProgressSteps / 12);
  const oscillationSignal = Math.min(currentEpisode.oscillationCount, 8) / 8;
  const recentAction = oneHotAction(currentEpisode.lastAction);
  const safePathExists = world.hasSafePlannedPath(personPos) ? 1 : 0;
  const currentDanger = world.dangerMap[personPos.r][personPos.c];
  const normalizedDanger = Number.isFinite(currentDanger) ? Math.max(0, 1 - Math.min(currentDanger, 10) / 10) : 0;
  const frontierDensity = Math.min(world.frontiers.size, 20) / 20;

  const features = [
    ...wallBits,
    ...oneHotVector(4, junctionType),
    ...directionOneHot(frontierInfo.direction),
    frontierInfo.density,
    world.exitSeen ? 1 : 0,
    ...directionOneHot(exitInfo.direction),
    exitInfo.desirability,
    monsterInfo.distance,
    ...directionOneHot(monsterInfo.direction),
    monsterInfo.threat,
    revisitPressure,
    novelty,
    oscillationSignal,
    safePathExists,
    frontierDensity,
    normalizedDanger,
    ...recentAction,
    ...exploredBits,
    ...unknownBits
  ];

  return features.slice(0, STATE_SIZE);
}

function getJunctionType(openCount) {
  if (openCount <= 1) {
    return 0;
  }
  if (openCount === 2) {
    return 1;
  }
  if (openCount === 3) {
    return 2;
  }
  return 3;
}

function getFrontierInfo() {
  const path = currentEpisode.world.getBestSafeFrontierPath(personPos);
  if (path.length < 2) {
    return { direction: 'NONE', density: 0 };
  }
  return {
    direction: directionNameForStep(personPos, path[1]),
    density: Math.min(currentEpisode.world.safeFrontiers.size || currentEpisode.world.frontiers.size, 12) / 12
  };
}

function getExitInfo() {
  if (!currentEpisode.world.exitSeen || !currentEpisode.world.exitPosition) {
    return { direction: 'NONE', desirability: 0 };
  }
  const path = Planner.findPath(personPos, (node) => positionsMatch(node, currentEpisode.world.exitPosition), currentEpisode.world, 'exit');
  if (path.length < 2) {
    return { direction: 'NONE', desirability: 0 };
  }
  return {
    direction: directionNameForStep(personPos, path[1]),
    desirability: Math.max(0, 1 - Math.min(path.length, 20) / 20)
  };
}

function getMonsterInfo() {
  const maxDistance = activeRows + activeCols;
  const distance = Math.min(manhattan(personPos, monsterPos), maxDistance) / maxDistance;
  const danger = currentEpisode.world.dangerMap[personPos.r][personPos.c];
  return {
    distance,
    direction: relativeDirection(personPos, monsterPos),
    threat: Number.isFinite(danger) ? Math.max(0, 1 - Math.min(danger, 10) / 10) : 0
  };
}

function oneHotVector(size, hotIndex) {
  return Array.from({ length: size }, (_, index) => (index === hotIndex ? 1 : 0));
}

function oneHotAction(actionIndex) {
  return Array.from({ length: ACTIONS.length }, (_, index) => (index === actionIndex ? 1 : 0));
}

function directionOneHot(direction) {
  const values = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE'];
  return values.map((value) => (value === direction ? 1 : 0));
}

function directionNameForStep(start, next) {
  for (let index = 0; index < ACTIONS.length; index += 1) {
    const action = ACTIONS[index];
    if (start.r + action.dr === next.r && start.c + action.dc === next.c) {
      return action.name;
    }
  }
  return 'NONE';
}

function relativeDirection(from, to) {
  const dr = to.r - from.r;
  const dc = to.c - from.c;
  if (Math.abs(dr) > Math.abs(dc)) {
    return dr < 0 ? 'UP' : 'DOWN';
  }
  if (dc !== 0) {
    return dc < 0 ? 'LEFT' : 'RIGHT';
  }
  return 'NONE';
}

function maybeAdvanceCurriculum() {
  if (metrics.curriculumStageIndex >= CURRICULUM_STAGES.length - 1 || metrics.recentOutcomes.length < RECENT_WINDOW) {
    return;
  }

  if (getRecentWinRate() >= 55 && metrics.heldOutEvalWinRate >= 35) {
    metrics.curriculumStageIndex += 1;
    statusDisplay.style.color = 'var(--exit-color)';
    statusDisplay.textContent = `Curriculum advanced to stage ${CURRICULUM_STAGES[metrics.curriculumStageIndex].label}.`;
  }
}

function finishTrainingEpisode(outcome) {
  metrics.totalEpisodes += 1;
  metrics.lastRunType = 'training';
  metrics.totalCoverage += getCoverageRatio();
  recordGameOutcome(outcome.reason);
  metrics.recentOutcomes.push(outcome.reason === 'escaped' ? 1 : 0);
  if (metrics.recentOutcomes.length > RECENT_WINDOW) {
    metrics.recentOutcomes.shift();
  }

  if (outcome.reason === 'escaped') {
    metrics.trainingWins += 1;
    metrics.winEpisodes += 1;
    metrics.totalWinSteps += currentEpisode.stepCount;
  }
}

function getRecentWinRate() {
  if (metrics.recentOutcomes.length === 0) {
    return 0;
  }
  return (metrics.recentOutcomes.reduce((sum, value) => sum + value, 0) / metrics.recentOutcomes.length) * 100;
}

function generateMaze(stage, seed) {
  syncMazeDimensions(stage.rows, stage.cols);
  const rng = new RNG(seed);
  grid = Array.from({ length: activeRows }, () => Array(activeCols).fill(WALL));

  const stack = [{ r: 1, c: 1 }];
  grid[1][1] = PATH;
  const directions = [
    { dr: -2, dc: 0 },
    { dr: 2, dc: 0 },
    { dr: 0, dc: -2 },
    { dr: 0, dc: 2 }
  ];

  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    const options = [];
    for (const direction of directions) {
      const nr = current.r + direction.dr;
      const nc = current.c + direction.dc;
      if (nr > 0 && nr < activeRows - 1 && nc > 0 && nc < activeCols - 1 && grid[nr][nc] === WALL) {
        options.push({ r: nr, c: nc, wr: current.r + direction.dr / 2, wc: current.c + direction.dc / 2 });
      }
    }
    if (options.length === 0) {
      stack.pop();
      continue;
    }
    const next = options[rng.int(options.length)];
    grid[next.wr][next.wc] = PATH;
    grid[next.r][next.c] = PATH;
    stack.push({ r: next.r, c: next.c });
  }

  carveRooms(stage, rng);
  knockWalls(stage, rng);

  pSpawn = { r: 1, c: 1 };
  const spawnDistances = bfsDistances(pSpawn, grid, activeRows, activeCols, true);
  exitPos = pickFarthestReachable(spawnDistances, pSpawn);
  reachablePathCount = countReachablePathTiles(spawnDistances);
  stepLimit = Math.max(260, Math.floor(reachablePathCount * stage.stepMultiplier));
  mSpawn = pickMonsterSpawn(spawnDistances, exitPos, stage, rng);
  personPos = { ...pSpawn };
  monsterPos = { ...mSpawn };
}

function carveRooms(stage, rng) {
  for (let index = 0; index < stage.roomCount; index += 1) {
    const rSize = rng.int(stage.roomMax - stage.roomMin + 1) + stage.roomMin;
    const cSize = rng.int(stage.roomMax - stage.roomMin + 1) + stage.roomMin;
    const sr = rng.int(activeRows - rSize - 2) + 1;
    const sc = rng.int(activeCols - cSize - 2) + 1;
    for (let row = sr; row < sr + rSize; row += 1) {
      for (let col = sc; col < sc + cSize; col += 1) {
        grid[row][col] = PATH;
      }
    }
  }
}

function knockWalls(stage, rng) {
  const wallsToKnock = Math.floor(activeRows * activeCols * stage.wallKnockRatio);
  let knocked = 0;
  while (knocked < wallsToKnock) {
    const row = rng.int(activeRows - 2) + 1;
    const col = rng.int(activeCols - 2) + 1;
    if (grid[row][col] !== WALL) {
      continue;
    }

    let pathNeighbors = 0;
    if (grid[row - 1][col] === PATH) pathNeighbors += 1;
    if (grid[row + 1][col] === PATH) pathNeighbors += 1;
    if (grid[row][col - 1] === PATH) pathNeighbors += 1;
    if (grid[row][col + 1] === PATH) pathNeighbors += 1;

    if (pathNeighbors >= 2) {
      grid[row][col] = PATH;
      knocked += 1;
    }
  }
}

function buildMazeDom() {
  mazeContainer.innerHTML = '';
  cells = Array.from({ length: activeRows }, () => Array(activeCols).fill(null));
  for (let row = 0; row < activeRows; row += 1) {
    for (let col = 0; col < activeCols; col += 1) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      mazeContainer.appendChild(cell);
      cells[row][col] = cell;
    }
  }
}

function syncMazeDom() {
  for (let row = 0; row < activeRows; row += 1) {
    for (let col = 0; col < activeCols; col += 1) {
      const cell = cells[row][col];
      cell.className = 'cell';
      if (grid[row][col] === PATH) {
        cell.classList.add('path');
      }
      if (row === exitPos.r && col === exitPos.c) {
        cell.classList.add('exit');
      }
    }
  }
}

function resetEpisode() {
  personPos = { ...pSpawn };
  monsterPos = { ...mSpawn };
  currentEpisode = createEpisodeState();
  currentEpisode.visitedCounts.set(keyFor(personPos), 1);
  rememberPathVisit(personPos);
  currentEpisode.world.observe(personPos, grid, exitPos, monsterPos, 0);
}

function rememberPathVisit(position) {
  const key = keyFor(position);
  currentEpisode.heatVisits.set(key, (currentEpisode.heatVisits.get(key) || 0) + 1);
}

function drawGrid() {
  for (let row = 0; row < activeRows; row += 1) {
    for (let col = 0; col < activeCols; col += 1) {
      const key = `${row}_${col}`;
      const cell = cells[row][col];
      cell.classList.toggle('known', currentEpisode.world.explored.has(key) || currentEpisode.world.occupancy[row][col] === 'wall');
      cell.classList.toggle('hot', (currentEpisode.heatVisits.get(key) || 0) >= 4 && grid[row][col] === PATH);
    }
  }

  document.querySelectorAll('.person, .monster').forEach((entity) => entity.remove());
  const personEl = document.createElement('div');
  personEl.className = 'person';
  cells[personPos.r][personPos.c].appendChild(personEl);
  const monsterEl = document.createElement('div');
  monsterEl.className = 'monster';
  cells[monsterPos.r][monsterPos.c].appendChild(monsterEl);
}

function shouldMonsterMove() {
  return currentEpisode.stepCount >= currentMazeStage.monsterWarmupSteps && currentEpisode.stepCount % currentMazeStage.monsterMoveEvery === 0;
}

function moveMonsterGlobal() {
  const distances = bfsDistances(personPos, grid, activeRows, activeCols, true);
  let bestStep = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const action of ACTIONS) {
    const next = { r: monsterPos.r + action.dr, c: monsterPos.c + action.dc };
    if (!isPath(next.r, next.c)) {
      continue;
    }
    const distance = distances[next.r][next.c];
    if (distance < bestDistance) {
      bestDistance = distance;
      bestStep = next;
    }
  }
  if (bestStep) {
    monsterPos = bestStep;
  }
}

function bfsDistances(start, gridRef, rows, cols, pathOnly) {
  const distances = Array.from({ length: rows }, () => Array(cols).fill(Number.POSITIVE_INFINITY));
  const queue = [{ ...start }];
  distances[start.r][start.c] = 0;
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    const currentDistance = distances[current.r][current.c];
    for (const action of ACTIONS) {
      const nextRow = current.r + action.dr;
      const nextCol = current.c + action.dc;
      if (!isInside(nextRow, nextCol)) {
        continue;
      }
      if (pathOnly && gridRef[nextRow][nextCol] !== PATH) {
        continue;
      }
      if (distances[nextRow][nextCol] !== Number.POSITIVE_INFINITY) {
        continue;
      }
      distances[nextRow][nextCol] = currentDistance + 1;
      queue.push({ r: nextRow, c: nextCol });
    }
  }
  return distances;
}

function pickFarthestReachable(distances, fallback) {
  let best = { ...fallback };
  let bestDistance = -1;
  for (let row = 1; row < activeRows - 1; row += 1) {
    for (let col = 1; col < activeCols - 1; col += 1) {
      const distance = distances[row][col];
      if (grid[row][col] === PATH && Number.isFinite(distance) && distance > bestDistance) {
        bestDistance = distance;
        best = { r: row, c: col };
      }
    }
  }
  return best;
}

function pickMonsterSpawn(distances, forbiddenTarget, stage, rng) {
  const threshold = Math.floor(getMaxDistance(distances) * stage.monsterSpawnThreshold);
  const candidates = [];
  for (let row = 1; row < activeRows - 1; row += 1) {
    for (let col = 1; col < activeCols - 1; col += 1) {
      const distance = distances[row][col];
      if (grid[row][col] === PATH && Number.isFinite(distance) && distance >= threshold && !(row === forbiddenTarget.r && col === forbiddenTarget.c)) {
        candidates.push({ r: row, c: col });
      }
    }
  }
  return candidates.length > 0 ? candidates[rng.int(candidates.length)] : { ...pSpawn };
}

function getMaxDistance(distances) {
  let maxDistance = 0;
  for (let row = 0; row < activeRows; row += 1) {
    for (let col = 0; col < activeCols; col += 1) {
      if (Number.isFinite(distances[row][col])) {
        maxDistance = Math.max(maxDistance, distances[row][col]);
      }
    }
  }
  return maxDistance;
}

function countReachablePathTiles(distances) {
  let count = 0;
  for (let row = 0; row < activeRows; row += 1) {
    for (let col = 0; col < activeCols; col += 1) {
      if (grid[row][col] === PATH && Number.isFinite(distances[row][col])) {
        count += 1;
      }
    }
  }
  return count;
}

function isDeadEnd(position) {
  let exits = 0;
  for (const action of ACTIONS) {
    if (isPath(position.r + action.dr, position.c + action.dc)) {
      exits += 1;
    }
  }
  return exits <= 1;
}

function getCoverageRatio() {
  return reachablePathCount === 0 ? 0 : currentEpisode.world.explored.size / reachablePathCount;
}

function syncMazeDimensions(rows, cols) {
  if (rows === activeRows && cols === activeCols && cells.length > 0) {
    return;
  }

  activeRows = rows;
  activeCols = cols;
  mazeContainer.style.gridTemplateColumns = `repeat(${activeCols}, var(--cell-size))`;
  buildMazeDom();
}

function setActiveTab(panelId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabTarget === panelId;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-selected', String(isActive));
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.id === panelId;
    panel.classList.toggle('active', isActive);
    panel.setAttribute('aria-hidden', String(!isActive));
  });
}

function recordGameOutcome(reason) {
  if (reason === 'running') {
    return;
  }

  const won = reason === 'escaped';
  metrics.gamesPlayed += 1;
  if (won) {
    metrics.gamesWon += 1;
  }
  metrics.gameOutcomes.push(won ? 1 : 0);
  if (metrics.gameOutcomes.length > RECENT_WINDOW) {
    metrics.gameOutcomes.shift();
  }
}

function getOverallWinRate() {
  if (metrics.gamesPlayed === 0) {
    return 0;
  }
  return (metrics.gamesWon / metrics.gamesPlayed) * 100;
}

function getRecentGameWinRate() {
  if (metrics.gameOutcomes.length === 0) {
    return 0;
  }
  return (metrics.gameOutcomes.reduce((sum, value) => sum + value, 0) / metrics.gameOutcomes.length) * 100;
}

function isOppositeAction(first, second) {
  return (
    (first === 0 && second === 1) ||
    (first === 1 && second === 0) ||
    (first === 2 && second === 3) ||
    (first === 3 && second === 2)
  );
}

function actionIndexForMove(start, next) {
  for (let index = 0; index < ACTIONS.length; index += 1) {
    const action = ACTIONS[index];
    if (start.r + action.dr === next.r && start.c + action.dc === next.c) {
      return index;
    }
  }
  return 0;
}

function updateStats(modeOverride) {
  const mode = modeOverride || runMode;
  const liveRows = mode === 'train' && workerTrainingState ? workerTrainingState.activeRows : activeRows;
  const liveCols = mode === 'train' && workerTrainingState ? workerTrainingState.activeCols : activeCols;
  const liveStepCount = mode === 'train' && workerTrainingState ? workerTrainingState.currentStepCount : currentEpisode.stepCount;
  const liveStepLimit = mode === 'train' && workerTrainingState ? workerTrainingState.stepLimit : stepLimit;
  const liveCoverage = mode === 'train' && workerTrainingState ? workerTrainingState.currentCoverage : getCoverageRatio() * 100;
  const liveParamCount = mode === 'train' && workerTrainingState ? workerTrainingState.paramCount : (agent ? agent.model.countParams() : 0);
  const trainingWinRate = metrics.totalEpisodes === 0 ? 0 : (metrics.trainingWins / metrics.totalEpisodes) * 100;
  const recentWinRate = getRecentWinRate();
  const overallWinRate = getOverallWinRate();
  const overallLossRate = metrics.gamesPlayed === 0 ? 0 : 100 - overallWinRate;
  const recentGameRate = getRecentGameWinRate();
  const averageCoverage = metrics.totalEpisodes === 0 ? 0 : (metrics.totalCoverage / metrics.totalEpisodes) * 100;
  const averageStepsOnWins = metrics.winEpisodes === 0 ? 0 : metrics.totalWinSteps / metrics.winEpisodes;
  const currentMazeLabel = `${liveRows} x ${liveCols}`;
  const isSmallerTrainingBoard = liveRows < MAX_ROWS || liveCols < MAX_COLS;

  metricMazeSize.textContent = currentMazeLabel;
  metricMazeNote.textContent = isSmallerTrainingBoard
    ? 'Smaller board for pattern learning before scale-up'
    : 'Full-size board for transfer evaluation';
  metricGamesPlayed.textContent = `${metrics.gamesPlayed}`;
  metricWinLoss.textContent = `${overallWinRate.toFixed(1)}% / ${overallLossRate.toFixed(1)}%`;
  metricWinLossNote.textContent = `${metrics.gamesWon} wins, ${Math.max(0, metrics.gamesPlayed - metrics.gamesWon)} losses`;
  metricLast25.textContent = `${recentGameRate.toFixed(1)}%`;

  detailEpisodes.textContent = `${metrics.totalEpisodes}`;
  detailWins.textContent = `${trainingWinRate.toFixed(1)}%`;
  detailRecent.textContent = `${recentWinRate.toFixed(1)}%`;
  detailEval.textContent = `${metrics.heldOutEvalWinRate.toFixed(1)}%`;
  detailStage.textContent = CURRICULUM_STAGES[metrics.curriculumStageIndex].label;
  detailSteps.textContent = metrics.winEpisodes === 0 ? '0' : averageStepsOnWins.toFixed(1);
  detailCoverage.textContent = `${averageCoverage.toFixed(1)}%`;
  detailModel.textContent = `${liveParamCount}`;
  detailLastLoss.textContent = formatLossReason(metrics.lastLossReason);
  detailMazeSize.textContent = currentMazeLabel;
  detailEvalSummary.textContent = metrics.lastEvalSummary;
  detailDeaths.textContent = `${metrics.monsterDeaths}`;
  detailStalls.textContent = `${metrics.stallFailures}`;
  detailSource.textContent = metrics.lastActionSource;
  detailRunType.textContent = metrics.lastRunType;
  detailCheckpoint.textContent = metrics.bestCheckpointImproved ? 'yes' : 'no';
  detailDiscoveries.textContent = `${metrics.frontierDiscoveries}`;
  detailFrontier.textContent = `${metrics.safeFrontierPicks}`;
  detailRepeat.textContent = `${metrics.loopRecoveries}`;
  detailOscillation.textContent = `${metrics.oscillationFailures}`;

  const modeLabel = mode === 'train' ? 'Training' : mode === 'watch' ? 'Watching' : mode === 'eval' ? 'Evaluating' : 'Idle';
  const watchLabel = visualEpisodeNumber ? ` | watching ep ${visualEpisodeNumber}` : '';
  const bestEval = metrics.bestCheckpointEpisode > 0
    ? ` | best eval ${metrics.bestEvalWinRate.toFixed(1)}% @ ep ${metrics.bestCheckpointEpisode}`
    : '';
  const speedLabel = mode === 'watch' ? ` | watch x${WATCH_STEPS_PER_TICK}` : '';
  statsDisplay.textContent = `Episode: ${metrics.totalEpisodes} | Mode: ${modeLabel} | epsilon ${epsilon.toFixed(3)} | steps ${liveStepCount}/${liveStepLimit} | current coverage ${liveCoverage.toFixed(1)}%${watchLabel}${bestEval}${speedLabel}`;
  trainingSummary.textContent = `Training starts on ${CURRICULUM_STAGES[0].rows} x ${CURRICULUM_STAGES[0].cols} mazes to learn reusable patterns, then scales toward ${CURRICULUM_STAGES[CURRICULUM_STAGES.length - 1].rows} x ${CURRICULUM_STAGES[CURRICULUM_STAGES.length - 1].cols}. Current board: ${currentMazeLabel}.`;
}

function syncConfiguredRunsFromInputs() {
  metrics.configuredTrainEpisodes = clampRunCount(trainEpisodesInput.value, DEFAULT_TRAIN_EPISODES_PER_RUN);
  metrics.configuredEvalEpisodes = clampRunCount(evalEpisodesInput.value, DEFAULT_EVAL_EPISODES);
  trainEpisodesInput.value = String(metrics.configuredTrainEpisodes);
  evalEpisodesInput.value = String(metrics.configuredEvalEpisodes);
}

function syncInputsFromMetrics() {
  trainEpisodesInput.value = String(metrics.configuredTrainEpisodes || DEFAULT_TRAIN_EPISODES_PER_RUN);
  evalEpisodesInput.value = String(metrics.configuredEvalEpisodes || DEFAULT_EVAL_EPISODES);
  stopBtn.disabled = true;
}

function clampRunCount(rawValue, fallback) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(1, Math.min(5000, Math.floor(parsed)));
}

function formatLossReason(reason) {
  const labels = {
    none: 'none',
    caught: 'caught',
    stall: 'stall',
    oscillation: 'oscillation',
    timeout: 'timeout'
  };
  return labels[reason] || reason;
}

function formatEvalSummary(reasonCounts, completedEpisodes) {
  if (completedEpisodes === 0) {
    return 'none';
  }

  const segments = [];
  if (reasonCounts.escaped > 0) {
    segments.push(`win ${reasonCounts.escaped}`);
  }
  if (reasonCounts.caught > 0) {
    segments.push(`caught ${reasonCounts.caught}`);
  }
  if (reasonCounts.stall > 0) {
    segments.push(`stall ${reasonCounts.stall}`);
  }
  if (reasonCounts.oscillation > 0) {
    segments.push(`osc ${reasonCounts.oscillation}`);
  }
  if (reasonCounts.timeout > 0) {
    segments.push(`timeout ${reasonCounts.timeout}`);
  }

  return segments.length === 0 ? 'none' : segments.join(', ');
}

function positionsMatch(first, second) {
  return first.r === second.r && first.c === second.c;
}

function manhattan(first, second) {
  return Math.abs(first.r - second.r) + Math.abs(first.c - second.c);
}

function isInside(row, col) {
  return row >= 0 && row < activeRows && col >= 0 && col < activeCols;
}

function isPath(row, col) {
  return isInside(row, col) && grid[row][col] === PATH;
}

function keyFor(position) {
  return `${position.r}_${position.c}`;
}

async function saveProgress() {
  if (!agent) {
    return;
  }
  await agent.saveLatest();
  persistProgressSnapshot(createProgressSnapshot());
}

async function loadProgress() {
  const raw = localStorage.getItem(STORAGE_META_KEY);
  if (!agent) {
    return;
  }

  if (!raw) {
    await hydrateAgentModels();
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    metrics = { ...createMetrics(), ...parsed.metrics };
    epsilon = parsed.epsilon ?? 1;
    agent.restoreMeta(parsed.agentMeta);
    await hydrateAgentModels();
  } catch {
    localStorage.removeItem(STORAGE_META_KEY);
  }
}

async function clearProgress() {
  localStorage.removeItem(STORAGE_META_KEY);
  await tf.io.removeModel(STORAGE_LATEST_MODEL).catch(() => {});
  await tf.io.removeModel(STORAGE_BEST_MODEL).catch(() => {});
  await tf.io.removeModel(LEGACY_STORAGE_LATEST_MODEL).catch(() => {});
  await tf.io.removeModel(LEGACY_STORAGE_BEST_MODEL).catch(() => {});
}

function createProgressSnapshot() {
  return {
    metrics,
    epsilon,
    agentMeta: agent ? agent.getMeta() : null
  };
}

function persistProgressSnapshot(snapshot) {
  localStorage.setItem(STORAGE_META_KEY, JSON.stringify({
    metrics: snapshot.metrics,
    epsilon: snapshot.epsilon,
    agentMeta: snapshot.agentMeta
  }));
}

async function hydrateAgentModels() {
  const latestLoaded = await agent.loadLatest();
  if (!latestLoaded) {
    await migrateLegacyModel(LEGACY_STORAGE_LATEST_MODEL, 'latest');
  }

  const bestLoaded = await agent.loadBest();
  if (!bestLoaded) {
    await migrateLegacyModel(LEGACY_STORAGE_BEST_MODEL, 'best');
  }
}

async function migrateLegacyModel(storageKey, target) {
  try {
    const legacyModel = await tf.loadLayersModel(storageKey);
    legacyModel.compile({ optimizer: tf.train.adam(agent.learningRate), loss: tf.losses.huberLoss });

    if (target === 'latest') {
      agent.model.dispose();
      agent.model = legacyModel;
      agent.targetModel.dispose();
      agent.bestModel.dispose();
      agent.targetModel = agent.createModel();
      agent.bestModel = agent.createModel();
      agent.syncTargetModel();
      agent.syncBestModel();
      await agent.saveLatest();
      return true;
    }

    agent.bestModel.dispose();
    agent.bestModel = legacyModel;
    await agent.saveBest();
    return true;
  } catch {
    return false;
  }
}

async function refreshMainAgentModels() {
  if (!agent) {
    return;
  }
  await agent.loadLatest();
  await agent.loadBest();
}

async function initializeTrainingWorker() {
  if (!window.Worker) {
    return;
  }

  if (trainingWorker) {
    trainingWorker.terminate();
  }

  workerReady = false;
  await new Promise((resolve) => {
    const workerUrl = new URL('training-worker.js', window.location.href);
    trainingWorker = new Worker(workerUrl);

    const timeoutId = window.setTimeout(() => {
      if (!workerReady && trainingWorker) {
        trainingWorker.terminate();
        trainingWorker = null;
        resolve();
      }
    }, 2000);

    const onError = () => {
      clearTimeout(timeoutId);
      workerReady = false;
      if (trainingWorker) {
        trainingWorker.terminate();
        trainingWorker = null;
      }
      resolve();
    };

    trainingWorker.addEventListener('message', (event) => {
      void handleTrainingWorkerMessage(event);
    });
    trainingWorker.addEventListener('error', onError, { once: true });

    const onReady = (event) => {
      if (event.data?.type === 'ready') {
        clearTimeout(timeoutId);
        trainingWorker.removeEventListener('message', onReady);
        resolve();
      }
    };
    trainingWorker.addEventListener('message', onReady);
    trainingWorker.postMessage({ type: 'init', payload: createProgressSnapshot() });
  });
}

function applyWorkerSnapshot(snapshot) {
  metrics = { ...createMetrics(), ...snapshot.metrics };
  epsilon = snapshot.epsilon ?? epsilon;
  workerTrainingState = {
    activeRows: snapshot.activeRows,
    activeCols: snapshot.activeCols,
    stepLimit: snapshot.stepLimit,
    currentCoverage: snapshot.currentCoverage,
    currentStepCount: snapshot.currentStepCount,
    paramCount: snapshot.paramCount
  };
}

async function handleTrainingWorkerMessage(event) {
  const { type, payload } = event.data || {};

  if (type === 'ready') {
    workerReady = true;
    if (payload) {
      applyWorkerSnapshot(payload);
    }
    return;
  }

  if (type === 'progress') {
    applyWorkerSnapshot(payload);
    statusDisplay.style.color = 'var(--text-color)';
    if (payload.statusText) {
      statusDisplay.textContent = payload.statusText;
    }
    updateStats('train');
    return;
  }

  if (type === 'persist') {
    applyWorkerSnapshot(payload);
    persistProgressSnapshot(payload);
    if (payload.statusText) {
      statusDisplay.textContent = payload.statusText;
    }
    updateStats(runMode === 'train' ? 'train' : 'idle');
    return;
  }

  if (type === 'completed') {
    applyWorkerSnapshot(payload);
    persistProgressSnapshot(payload);
    await refreshMainAgentModels();
    currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
    generateMaze(currentMazeStage, TRAIN_SEED_BASE + metrics.totalEpisodes * 7919 + 3);
    syncMazeDom();
    resetEpisode();
    drawGrid();
    statusDisplay.style.color = stopRequested ? 'var(--text-color)' : 'var(--exit-color)';
    if (payload.statusText) {
      statusDisplay.textContent = payload.statusText;
    }
    stopSimulation();
    return;
  }

  if (type === 'error') {
    statusDisplay.style.color = 'var(--monster-color)';
    statusDisplay.textContent = payload?.message || 'Training worker failed.';
    stopSimulation();
  }
}

function runSelfChecks() {
  const vector = encodeStateFeatures();
  console.assert(vector.length === STATE_SIZE, 'State vector size mismatch');
  const buffer = new PrioritizedReplayBuffer(3);
  buffer.push({ id: 1 }, 1);
  buffer.push({ id: 2 }, 2);
  buffer.push({ id: 3 }, 3);
  buffer.push({ id: 4 }, 4);
  console.assert(buffer.size === 3, 'Replay buffer capacity mismatch');
  const normalizer = new RewardNormalizer();
  normalizer.observe(10);
  console.assert(normalizer.normalize(10) <= 4, 'Reward normalization mismatch');
}

function argMax(values) {
  let bestIndex = 0;
  let bestValue = values[0];
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

document.addEventListener('DOMContentLoaded', () => {
  init().catch((error) => {
    console.error(error);
    statusDisplay.style.color = 'var(--monster-color)';
    statusDisplay.textContent = 'Initialization failed. Check the console for details.';
  });
});