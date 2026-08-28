---
title: "The Latent Field"
created: 2025-12-06
published: 2025-12-06
modified: 2026-08-28
cssclasses:
  - home-page
---

<div style="position: relative; overflow: hidden; padding: 2rem 1.45rem 2.15rem; border: 1px solid var(--lightgray); border-radius: 24px; background: linear-gradient(135deg, rgba(132, 165, 157, .16), rgba(255, 178, 102, .12)); margin: 1rem 0 1.8rem; box-shadow: 0 18px 45px rgba(40, 75, 99, .08);">
  <div style="font-family: 'Schibsted Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif; font-size: 2rem; line-height: 1; margin-bottom: .8rem; letter-spacing: .01em;">🍊 The Latent Field</div>
  <p style="font-family: 'Schibsted Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif; letter-spacing: .18em; text-transform: uppercase; font-size: 1rem; line-height: 1.45; color: var(--secondary); margin: 0 0 .8rem;">Large Model Research System</p>
  <h1 style="font-family: KaiTi, STKaiti, serif; font-size: 2rem; line-height: 1.18; margin: 0 0 .9rem; font-weight: 700;">把大模型研究，展开成一张持续生长的地图</h1>
  <p style="font-family: KaiTi, STKaiti, serif; font-size: 1.16rem; line-height: 1.85; margin: 0 0 .8rem; color: var(--darkgray); max-width: 50rem;">大模型正在重塑我们理解智能、训练模型与构建系统的方式，也重新定义着研究本身的边界。论文提出新的可能，代码赋予它具体的形态，实验留下证据，工程实践则不断修正我们对方法与边界的认识。</p>
  <p style="font-family: KaiTi, STKaiti, serif; font-size: 1.16rem; line-height: 1.85; margin: 0; color: var(--darkgray); max-width: 50rem;">这里沿着<strong>模型、训练、推理、Agent 与系统</strong>展开，收集那些值得被留下的研究线索：已经得到验证的结论，实验中反复出现的现象，真实研发积累的经验，以及仍然没有答案的问题。<strong>因为研究从来不是答案的陈列，而是理解不断形成、被检验，再被重新书写的过程。</strong></p>
</div>

## Knowledge Architecture

知识节点沿着问题之间的依赖关系展开，而不是按照热点堆叠。这里从学习与表示的基础出发，经过模型结构、训练过程与推理系统，最终抵达工具、Agent 和真实任务；每一层既回答自身的问题，也为下一层提供理解的起点。

<div class="home-domain-rail">
  <a href="fundamentals/" class="home-domain-node internal"><span class="home-domain-number">01</span><strong>Fundamentals</strong><small>理解学习如何发生，以及能力如何形成</small><span class="home-domain-arrow">-&gt;</span></a>
  <a href="architecture/" class="home-domain-node internal"><span class="home-domain-number">02</span><strong>Architecture</strong><small>追踪表示、结构与规模化之间的关系</small><span class="home-domain-arrow">-&gt;</span></a>
  <a href="training/" class="home-domain-node internal"><span class="home-domain-number">03</span><strong>Training</strong><small>观察数据、目标与系统如何塑造能力</small><span class="home-domain-arrow">-&gt;</span></a>
  <a href="inference/" class="home-domain-node internal"><span class="home-domain-number">04</span><strong>Inference</strong><small>理解能力如何在推理时被调用与释放</small><span class="home-domain-arrow">-&gt;</span></a>
  <a href="application/" class="home-domain-node internal"><span class="home-domain-number">05</span><strong>Application</strong><small>考察模型如何进入任务、工具与环境</small><span class="home-domain-arrow">-&gt;</span></a>
</div>

## Research Workbench

研究材料先在工作台中保留它们各自的语境，再在相互对照中逐渐形成判断。论文提出问题与方法，源码展示方法如何成为系统，实验检验假设在具体条件下是否成立，工程记录则补足真实约束与反复试错留下的经验。

<div class="home-workbench-grid">
  <a href="sources/" class="home-workbench-item internal">
    <span class="home-workbench-label">01 / Sources</span>
    <strong>Research Notes</strong>
    <small>从论文、课程与技术文档中提取问题、方法、证据与边界。</small>
    <span class="home-item-arrow">-&gt;</span>
  </a>
  <a href="projects/" class="home-workbench-item internal">
    <span class="home-workbench-label">02 / Projects</span>
    <strong>Engineering Records</strong>
    <small>记录源码、复现实验、部署实践与真实系统中的问题。</small>
    <span class="home-item-arrow">-&gt;</span>
  </a>
</div>

## Learning Routes

学习路线不是必须线性完成的课程，而是几条可以往返的理解路径。面对一个新的现象，可以向下追溯它所依赖的机制，也可以从基础概念出发，沿着模型、训练与系统逐步观察能力如何显现。

<div class="home-route-grid">
  <a href="fundamentals/" class="home-route-card internal"><span class="home-route-index">01</span><span><strong>Foundations of Learning Systems</strong><small>Linear Algebra -> Probability -> Information Theory -> Optimization -> Learning Dynamics</small></span></a>
  <a href="architecture/" class="home-route-card internal"><span class="home-route-index">02</span><span><strong>Model Architecture</strong><small>Tokenization -> Transformer -> Attention -> Position -> Scaling</small></span></a>
  <a href="training/" class="home-route-card internal"><span class="home-route-index">03</span><span><strong>Training Systems</strong><small>Objective -> Data -> Parallelism -> Checkpoint -> Evaluation</small></span></a>
  <a href="inference/" class="home-route-card internal"><span class="home-route-index">04</span><span><strong>Inference and Agents</strong><small>Decoding -> KV Cache -> Serving -> Retrieval -> Tools -> Agents</small></span></a>
</div>

## Reading Protocol

阅读的终点不是复述材料，而是留下能够被重新检查的判断：问题从哪里开始，证据如何支持结论，结论又在哪些条件下需要被修正。

<div class="home-use-grid">
  <div><span class="home-use-index">LOCATE</span><p>从问题或任务出发，界定需要理解的机制与证据范围，先找到真正值得追问的地方。</p></div>
  <div><span class="home-use-index">RELATE</span><p>将新材料放回知识体系，与相关理论、源码和实践记录建立可以回溯的联系。</p></div>
  <div><span class="home-use-index">SYNTHESIZE</span><p>提炼经过验证的机制、结论与适用边界，让一次阅读成为后续研究可以继续调用的认识。</p></div>
</div>

<p class="home-footer-note">A living knowledge system for understanding large models from first principles to production systems.</p>
