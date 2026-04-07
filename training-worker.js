importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

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
const STORAGE_LATEST_MODEL = 'indexeddb://maze-hybrid-latest';
const STORAGE_BEST_MODEL = 'indexeddb://maze-hybrid-best';
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
let isTraining = false;
let stopRequested = false;
let epsilon = 1;
const minEpsilon = 0.05;
const epsilonDecay = 0.995;

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

function buildSnapshot(statusText = '') {
  return {
    metrics,
    epsilon,
    agentMeta: agent ? agent.getMeta() : null,
    activeRows,
    activeCols,
    stepLimit,
    currentCoverage: getCoverageRatio() * 100,
    currentStepCount: currentEpisode ? currentEpisode.stepCount : 0,
    paramCount: agent ? agent.model.countParams() : 0,
    statusText
  };
}

function postState(type, statusText = '') {
  postMessage({ type, payload: buildSnapshot(statusText) });
}

async function initWorker(payload) {
  if (agent) {
    await agent.dispose();
  }

  agent = new DQNAgent(STATE_SIZE, ACTIONS.length);
  metrics = { ...createMetrics(), ...(payload?.metrics || {}) };
  epsilon = payload?.epsilon ?? 1;
  agent.restoreMeta(payload?.agentMeta || null);
  await agent.loadLatest();
  await agent.loadBest();

  currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
  generateMaze(currentMazeStage, TRAIN_SEED_BASE + metrics.totalEpisodes * 7919 + 3);
  resetEpisode();
  postState('ready', 'Training worker ready.');
}

async function resetWorker() {
  stopRequested = true;
  isTraining = false;
  if (agent) {
    await agent.dispose();
  }
  agent = new DQNAgent(STATE_SIZE, ACTIONS.length);
  metrics = createMetrics();
  epsilon = 1;
  currentMazeStage = CURRICULUM_STAGES[0];
  generateMaze(currentMazeStage, TRAIN_SEED_BASE);
  resetEpisode();
  postState('ready', 'Training worker reset.');
}

async function startTraining(payload) {
  if (!agent || isTraining) {
    return;
  }

  metrics.configuredTrainEpisodes = payload?.trainEpisodes || metrics.configuredTrainEpisodes || DEFAULT_TRAIN_EPISODES_PER_RUN;
  metrics.configuredEvalEpisodes = payload?.evalEpisodes || metrics.configuredEvalEpisodes || DEFAULT_EVAL_EPISODES;
  stopRequested = false;
  isTraining = true;
  metrics.lastRunType = 'training';
  metrics.bestCheckpointImproved = false;
  postState('progress', `Training hybrid agent across fresh mazes for ${metrics.configuredTrainEpisodes} episodes.`);

  try {
    const targetEpisode = metrics.totalEpisodes + metrics.configuredTrainEpisodes;

    while (isTraining && !stopRequested && metrics.totalEpisodes < targetEpisode) {
      maybeAdvanceCurriculum();
      currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
      const episodeNumber = metrics.totalEpisodes + 1;
      const seed = TRAIN_SEED_BASE + episodeNumber * 7919;
      generateMaze(currentMazeStage, seed);
      resetEpisode();
      runHeadlessTrainingEpisode();

      if (metrics.totalEpisodes % REPLAY_INTERVAL === 0) {
        await agent.replay(REPLAY_BATCH_SIZE);
      }
      epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);

      if (metrics.totalEpisodes % EVAL_INTERVAL === 0) {
        await runHeldOutEvaluation(BACKGROUND_EVAL_EPISODES);
        await agent.saveLatest();
        postState('persist', `Background evaluation finished at ${metrics.heldOutEvalWinRate.toFixed(1)}%.`);
      } else if (metrics.totalEpisodes % SAVE_INTERVAL === 0) {
        await agent.saveLatest();
        postState('persist', 'Training checkpoint saved.');
      }

      if (metrics.totalEpisodes % UI_UPDATE_INTERVAL === 0) {
        postState('progress', `Training hybrid agent across fresh mazes for ${metrics.configuredTrainEpisodes} episodes.`);
        await tf.nextFrame();
      }
    }

    await agent.saveLatest();
    isTraining = false;
    postState('persist', stopRequested
      ? 'Training stopped. The latest checkpoint and metrics were saved.'
      : 'Training run finished. The best checkpoint is ready for unseen-maze evaluation.');
    postState('completed', stopRequested
      ? 'Training stopped. The latest checkpoint and metrics were saved.'
      : 'Training run finished. The best checkpoint is ready for unseen-maze evaluation.');
  } catch (error) {
    isTraining = false;
    postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error) } });
  }
}

function runHeadlessTrainingEpisode() {
  agent.startEpisode();
  while (true) {
    if (!isTraining || stopRequested) {
      break;
    }

    const state = encodeStateFeatures();
    const decision = chooseHybridAction(false, 'latest');
    const outcome = environmentStep(decision.action, decision.source);
    const nextState = encodeStateFeatures();
    agent.rememberTransition(state, decision.action, outcome.reward, nextState, outcome.done, decision.plannerAction, decision.plannerStrength);
    if (outcome.done) {
      finishTrainingEpisode(outcome);
      break;
    }
  }
}

async function runHeldOutEvaluation(episodeCount = metrics.configuredEvalEpisodes) {
  let wins = 0;
  let completedEpisodes = 0;

  for (let index = 0; index < episodeCount; index += 1) {
    if (stopRequested) {
      break;
    }
    generateMaze(CURRICULUM_STAGES[CURRICULUM_STAGES.length - 1], EVAL_SEEDS[index % EVAL_SEEDS.length]);
    resetEpisode();
    metrics.lastRunType = 'evaluation';
    completedEpisodes += 1;

    while (true) {
      if (stopRequested) {
        break;
      }
      const decision = chooseHybridAction(true, 'latest');
      const outcome = environmentStep(decision.action, decision.source);
      if (outcome.done) {
        recordGameOutcome(outcome.reason);
        if (outcome.reason === 'escaped') {
          wins += 1;
        }
        break;
      }
    }
  }

  metrics.heldOutEvalWinRate = completedEpisodes === 0 ? 0 : (wins / completedEpisodes) * 100;
  metrics.bestCheckpointImproved = false;
  if (!stopRequested && metrics.heldOutEvalWinRate >= metrics.bestEvalWinRate) {
    metrics.bestEvalWinRate = metrics.heldOutEvalWinRate;
    metrics.bestCheckpointEpisode = metrics.totalEpisodes;
    metrics.bestCheckpointImproved = true;
    agent.syncBestModel();
    await agent.saveBest();
  }

  currentMazeStage = CURRICULUM_STAGES[metrics.curriculumStageIndex];
  generateMaze(currentMazeStage, TRAIN_SEED_BASE + metrics.totalEpisodes * 7919 + 3);
  resetEpisode();
  metrics.lastRunType = 'training';
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

function environmentStep(actionIndex, source) {
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

function maybeAdvanceCurriculum() {
  if (metrics.curriculumStageIndex >= CURRICULUM_STAGES.length - 1 || metrics.recentOutcomes.length < RECENT_WINDOW) {
    return;
  }

  if (getRecentWinRate() >= 55 && metrics.heldOutEvalWinRate >= 35) {
    metrics.curriculumStageIndex += 1;
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
  activeRows = rows;
  activeCols = cols;
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

self.onmessage = async (event) => {
  const { type, payload } = event.data || {};

  try {
    if (type === 'init') {
      await initWorker(payload);
      return;
    }

    if (type === 'start-training') {
      await startTraining(payload);
      return;
    }

    if (type === 'stop') {
      stopRequested = true;
      return;
    }

    if (type === 'reset') {
      await resetWorker();
    }
  } catch (error) {
    postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error) } });
  }
};