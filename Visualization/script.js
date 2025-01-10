/***********************************************
 *  1) 全局变量 & 页面加载
 ***********************************************/
let data = [];         // 从 data.json 读取的所有回合数据
let currentIndex = 0;  // 当前回合索引
let currentView = "god"; // 当前视角: 'god' | 'player1' | 'player2'

window.addEventListener("DOMContentLoaded", () => {
  loadData();
});

/***********************************************
 *  2) 读取 data.json
 ***********************************************/
async function loadData() {
  try {
    const response = await fetch("VISdata.json");
    data = await response.json();

    // 设置滑杆最大值为 data 的长度-1
    const roundSlider = document.getElementById("roundSlider");
    roundSlider.max = data.length - 1;

    // 初始渲染第 0 回合
    updateRound(0);
  } catch (error) {
    console.error("加载 data.json 时出错：", error);
  }
}

/***********************************************
 *  3) 回合切换 / 视角切换
 ***********************************************/
// 滑杆变化时
function handleSliderChange(value) {
  currentIndex = parseInt(value, 10);
  updateRound(currentIndex);
  document.getElementById("roundIndicator").textContent = `第 ${currentIndex + 1} 回合`;
}

// 视角下拉框变化时
function handleViewChange(newView) {
  currentView = newView;
  updateRound(currentIndex);
}

/***********************************************
 *  4) updateRound() - 根据回合数据整体渲染
 ***********************************************/
function updateRound(index) {
  const roundData = data[index];
  if (!roundData) return;

  // 1. 更新牌堆
  renderPileState(roundData.pile_state);

  // 2. 更新玩家手牌
  const hands = roundData.hand_cards || [];
  // hands[0] => player1, hands[1] => player2
  renderPlayerHand("player1Cards", hands[0] || [], 0);
  renderPlayerHand("player2Cards", hands[1] || [], 1);

  // 3. 更新统计数据
  updateGameStats(roundData);


  // 4. Update discard pile
  renderDiscardPile(roundData.discard_pile || []);
}

/***********************************************
 *  4.1) updateGameStats() - 更新游戏统计数据
 ***********************************************/
function updateGameStats(roundData) {
  // 从 roundData 中提取数据
  const score = roundData.score || 0;
  const lives = roundData.life_tokens || 3;
  const hints = roundData.hints || 0;
  const totalCards = roundData.deck_size || 0;

  const action = roundData.action || "-";
  
  // 计算牌堆剩余数量（这里示例用所有玩家手牌数之和，你可以根据实际数据结构调整）
  //const totalCards = (roundData.hand_cards || []).reduce((sum, hand) => sum + (hand ? hand.length : 0), 0);
  
  // 更新显示
  document.getElementById('scoreDisplay').textContent = score;
  document.getElementById('livesDisplay').textContent = lives;
  document.getElementById('deckCountDisplay').textContent = totalCards;
  document.getElementById('hintsDisplay').textContent = hints;
  document.getElementById('actionDisplay').textContent = action;
}

/***********************************************
 *  5) 渲染牌堆 - 如果 pile_state[color] > 1, 显示对应颜色+数字，否则灰0
 ***********************************************/
function renderPileState(pileState) {
  const pileCardsContainer = document.getElementById("pileCards");
  pileCardsContainer.innerHTML = ""; // 清空

  // 五种颜色按顺序
  const colorOrder = ["yellow", "blue", "green", "red", "purple"];

  colorOrder.forEach((color) => {
    const value = pileState[color] || 0;
    let cardColorClass = "grey";
    let cardNumber = "0";
    if (value > 1) {
      cardColorClass = color;       // 原色
      cardNumber = String(value);
    }

    // 创建普通卡DOM
    const card = document.createElement("div");
    card.classList.add("card", cardColorClass);
    card.textContent = cardNumber;
    pileCardsContainer.appendChild(card);
  });
}

/***********************************************
 *  6) 渲染玩家手牌
 *  这里要考虑: cardData.type + currentView + playerIndex
 ***********************************************/
function renderPlayerHand(containerId, handCards, playerIndex) {
  const container = document.getElementById(containerId);
  container.innerHTML = ""; // 清空

  handCards.forEach((cardData) => {
    const { type } = cardData;

    if (type === "roulette") {
      // 这是轮盘卡
      renderRouletteCard(container, cardData, playerIndex);
    } else if (type === "normal") {
      // 这是普通卡
      renderNormalCard(container, cardData, playerIndex);
    } else {
      // 未知类型？ 也可以在 data.json 中都规定好，只是 normal/roulette
      // 这里简单跳过或渲染背面
      //renderBackside(container);
    }
  });
}

/***********************************************
 *  7) 渲染一张 “轮盘卡”
 ***********************************************/
function renderRouletteCard(container, cardData, playerIndex) {
  // cardData = { type:'roulette', color, number, possibleColors, possibleNumbers, ... }
  // 根据视角 & 玩家索引 判断:
  if (currentView === "god") {
    // 上帝视角 => 看真实的轮盘卡
    //createRouletteCard(container, cardData);

  } else if (currentView === "player1") {
    // 玩家1视角 => 只有 playerIndex===0 的牌 才显示轮盘卡，否则显示背面
    if (playerIndex === 0) {
      createRouletteCard(container, cardData);
    } else {
      //renderBackside(container);
    }

  } else if (currentView === "player2") {
    // 玩家2视角 => 只有 playerIndex===1 的牌 才显示轮盘卡，否则背面
    if (playerIndex === 1) {
      createRouletteCard(container, cardData);
    } else {
      //renderBackside(container);
    }
  }
}

/***********************************************
 *  8) 创建并追加一张 RouletteCard
 ***********************************************/
function createRouletteCard(container, cardData) {
  const { possibleColors = [], possibleNumbers = [] } = cardData;

  // 实例化 轮盘卡
  const rCard = new RouletteCard(possibleColors, possibleNumbers);
  // 也许你想让轮盘卡上再显示一下 color/number 也行，可再做扩展
  // e.g. rCard.element.title = `color=${cardData.color}, number=${cardData.number}`;

  // append
  rCard.appendTo(container);
}

/***********************************************
 *  9) 渲染一张 “普通卡”
 ***********************************************/
function renderNormalCard(container, cardData, playerIndex) {
  // cardData = { type:'normal', color, number, ... }
  // 这里示例：god / player1 / player2 都能直接看到对方的普通卡
  // 若你想实现“只有本人可见，对手看背面”，也可加判断
  // 根据视角 & 玩家索引 判断:
  if (currentView === "god") {
    // 上帝视角 => 看真实的轮盘卡
    //createRouletteCard(container, cardData);
    const { color, number } = cardData;
  const card = document.createElement("div");
  card.classList.add("card", color);
  card.textContent = number;
  container.appendChild(card);

  } else if (currentView === "player1") {
    // 玩家1视角 => 只有 playerIndex===0 的牌 才显示轮盘卡，否则显示背面
    if (playerIndex === 1) {
        const { color, number } = cardData;
        const card = document.createElement("div");
        card.classList.add("card", color);
        card.textContent = number;
        container.appendChild(card);
    } else {
      //renderBackside(container);
    }

  } else if (currentView === "player2") {
    // 玩家2视角 => 只有 playerIndex===1 的牌 才显示轮盘卡，否则背面
    if (playerIndex === 0) {
        const { color, number } = cardData;
  const card = document.createElement("div");
  card.classList.add("card", color);
  card.textContent = number;
  container.appendChild(card);
    } else {
      //renderBackside(container);
    }
  }
  
}

/***********************************************
 * 10) 渲染背面(不可见内容)
 ***********************************************/
function renderBackside(container) {
  const card = document.createElement("div");
  card.classList.add("card", "backside");
  container.appendChild(card);
}

/***********************************************
 * 11) 轮盘卡 类 (RouletteCard)
 ***********************************************/
class RouletteCard {
  /**
   * @param {string[]} possibleColors - 如 ["yellow","blue"]
   * @param {number[]} possibleNumbers - 如 [1,2,5]
   */
  constructor(possibleColors = [], possibleNumbers = []) {
    this.possibleColors = possibleColors;
    this.possibleNumbers = possibleNumbers;

    // 创建外壳 div，并添加公共 card 样式 + 轮盘专属样式
    this.element = document.createElement("div");
    this.element.classList.add("card", "roulette-card");

    // 让内部可以相对定位
    this.element.style.position = "relative";

    // 创建一个 canvas，用于绘制轮盘
    this.canvas = document.createElement("canvas");
    this.canvas.width = 100;
    this.canvas.height = 100;
    // 居中
    this.canvas.style.position = "absolute";
    this.canvas.style.top = "50%";
    this.canvas.style.left = "50%";
    this.canvas.style.transform = "translate(-50%, -50%)";
    this.canvas.style.backgroundColor = "transparent";

    this.element.appendChild(this.canvas);
    // 立即绘制轮盘
    this.drawRoulette();
  }

  appendTo(parent) {
    parent.appendChild(this.element);
  }

  drawRoulette() {
    const ctx = this.canvas.getContext("2d");
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) / 2;

    ctx.clearRect(0, 0, w, h);

    // 将 possibleColors / possibleNumbers 整合成 slots
    const slots = [];
    this.possibleColors.forEach((c) => {
      slots.push({ type: "color", value: c });
    });
    this.possibleNumbers.forEach((n) => {
      slots.push({ type: "number", value: n });
    });

    if (!slots.length) {
      // 无可绘制时，留白
      return;
    }

    const anglePerSlot = (2 * Math.PI) / slots.length;

    for (let i = 0; i < slots.length; i++) {
      const slot = slots[i];
      const startAngle = i * anglePerSlot;
      const endAngle = startAngle + anglePerSlot;

      let fillColor = "#777";
      let textToDraw = "";

      if (slot.type === "color") {
        fillColor = slot.value; // 直接用颜色字符串
      } else if (slot.type === "number") {
        // 简单映射: 1->a,2->b,3->c,4->d,5->e
        const mapping = {1:"1",2:"2",3:"3",4:"4",5:"5"};
        fillColor = "#555";
        textToDraw = mapping[slot.value] || "";
      }

      // 画扇形
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, startAngle, endAngle);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();

      // 写字
      if (textToDraw) {
        ctx.save();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 14px Arial";
        const midAngle = startAngle + anglePerSlot / 2;
        const textRadius = r * 0.65;
        const tx = cx + textRadius * Math.cos(midAngle);
        const ty = cy + textRadius * Math.sin(midAngle);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(textToDraw, tx, ty);
        ctx.restore();
      }
    }

    // 外圆描边
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#999";
    ctx.stroke();
  }
}

// Add new function to render discard pile
function renderDiscardPile(discardPile) {
  const container = document.getElementById("discardPileRight");
  container.innerHTML = ""; // Clear existing cards

  // Create a grid container for the cards
  const gridContainer = document.createElement("div");
  gridContainer.classList.add("discard-grid");
  
  // Sort cards by color and number for better organization
  const sortedCards = [...discardPile].sort((a, b) => {
    if (a.color !== b.color) {
      // First sort by color
      const colorOrder = ["red", "yellow", "green", "blue", "purple"];
      return colorOrder.indexOf(a.color) - colorOrder.indexOf(b.color);
    }
    // Then by number
    return a.number - b.number;
  });

  // Render each card in the discard pile
  sortedCards.forEach(cardData => {
    const card = document.createElement("div");
    card.classList.add("card", cardData.color);
    card.textContent = cardData.number;
    
    // Add a small label to show discard order if needed
    // const orderLabel = document.createElement("span");
    // orderLabel.classList.add("discard-order");
    // orderLabel.textContent = discardPile.indexOf(cardData) + 1;
    // card.appendChild(orderLabel);
    
    gridContainer.appendChild(card);
  });

  container.appendChild(gridContainer);
}