# CORTEX/1

```math
\mathcal{C}=\left(\vec{\iota},\vec{b},\vec{q},\vec{e}\right)
```

```math
\vec{\iota}=
\begin{bmatrix}\mathrm{numi\_lab}&\mathrm{codex\_continuity}&
\mathrm{apple\_native\_robotics\_learning}&\mathrm{ultimate\_physics\_learning\_system}&
\mathrm{compound\_platform\_advantage}&\mathrm{evidence\_over\_claims}
\end{bmatrix}
```

```math
\vec{b}=
\begin{bmatrix}b_0&b_1\end{bmatrix}=
\begin{bmatrix}\mathrm{production:numisolver}&\mathrm{archive:side}\end{bmatrix}
```

```math
\vec{q}=
\begin{bmatrix}
\mathrm{inspect\_owners}&\mathrm{trace\_live\_system}&\mathrm{change\_high\_leverage}&
\mathrm{verify\_proportional\_risk}&\mathrm{measure\_system\_outcome}&
\mathrm{compound\_advantage}&\mathrm{publish\_scoped}
\end{bmatrix},\qquad q_i\to q_{i+1}
```

```math
\vec{e}=
\begin{bmatrix}
\mathrm{code}&\mathrm{exact\_replay}&\mathrm{physical\_outcome}&
\mathrm{profiler}&\mathrm{test\_only}&\mathrm{plan\_only}
\end{bmatrix}
```

```math
\mathrm{shipped}\iff e_0\land\neg e_5,\qquad
\mathrm{proof}\not\Leftarrow e_4,\qquad
\mathrm{credibility}\propto e_1+e_2+e_3
```

## MAP

```math
\vec{s}=
\begin{bmatrix}
\mathrm{world}&\mathrm{metal}&\mathrm{visual}&\mathrm{tactile}&\mathrm{numeric}
\end{bmatrix}
```

```math
\vec{D}=
\begin{bmatrix}
\mathtt{docs/WORLD\_ENGINE.md}&
\mathtt{docs/METAL\_WORLD.md}&
\mathtt{docs/VISUAL\_PLATFORM.md}&
\mathtt{docs/TACTILE\_GEOMETRY\_BRIDGE.md}&
\mathtt{docs/NUMERICS.md}
\end{bmatrix},\qquad D_i\leftrightarrow s_i
```

```math
\mathrm{read}(\Delta)=D_{\mathrm{owner}(\Delta)}\to
\mathrm{code}_{\mathrm{live\_path}(\Delta)},\qquad
1\le\left|\vec{D}_{\mathrm{loaded}}\right|\le
\left|\mathrm{owners}(\Delta)\right|
```

## ARCH

```math
\vec{P}=
\begin{bmatrix}\mathrm{WorldPack}&\mathrm{TaskPack}&\mathrm{PolicyPack}&\mathrm{MotionPack?}\end{bmatrix}
```

```math
\vec{\Phi}_{\mathrm{physics}}=
\begin{bmatrix}
\mathrm{rigid}&\mathrm{articulated}&\mathrm{contact}&\mathrm{terrain}&
\mathrm{rod}&\mathrm{deformable}&\mathrm{tactile}&\mathrm{vision}
\end{bmatrix},\qquad
\lim_{t\to\infty}\vec{\Phi}_{\mathrm{production}}(t)=\vec{1}
```

```math
\mathrm{compile}(\vec{P}):
\vec{n}\to
\begin{bmatrix}
\mathrm{stable\_indices}&\mathrm{tables}&\mathrm{counts}&
\mathrm{capacities}&\mathrm{fingerprints}
\end{bmatrix}
```

```math
\vec{x}_{\mathrm{Swift}}=
\begin{bmatrix}
\mathrm{rollout}&\mathrm{cadence}&\mathrm{submission\_ring}&
\mathrm{completion}&\mathrm{timeout}&\mathrm{revision}
\end{bmatrix}
```

```math
\vec{x}_{\mathrm{Metal}}=
\begin{bmatrix}
\mathrm{physics}&\mathrm{contact}&\mathrm{terrain}&\mathrm{control}&\mathrm{rng}&
\mathrm{sense}&\mathrm{observe}&\mathrm{reward}&\mathrm{done}&\mathrm{reset}
\end{bmatrix}
```

```math
\vec{x}_{\mathrm{MLX}}=
\begin{bmatrix}\mathrm{learner}\end{bmatrix},\qquad
\vec{x}_{\mathrm{MLX}}\cap
\begin{bmatrix}\mathrm{physics}&\mathrm{simulator\_state}&\mathrm{rollout\_scheduler}\end{bmatrix}
=\varnothing
```

```math
\vec{S}_{\mathrm{persistent}}\subset\mathrm{device\_private},\qquad
\mathrm{publish}(\vec{S})=\mathrm{compact\_rollout}
```

```math
\vec{R}_{\mathrm{atomic}}=
\begin{bmatrix}
\mathrm{articulation}&\mathrm{scene}&\mathrm{contact}&\mathrm{warmstart}&
\mathrm{actuator}&\mathrm{sensor}&\mathrm{episode}&\mathrm{rng}
\end{bmatrix}
```

```math
\mathrm{robot}_{\mathrm{new}}=
\begin{bmatrix}\mathrm{mechanics}&\mathrm{authored\_packs}&\mathrm{policy\_contract}\end{bmatrix}
\not\ni
\begin{bmatrix}\mathrm{robot\_shader}&\mathrm{host\_mode}\end{bmatrix}
```

```math
\mathrm{visual}=\mathrm{authored\_presentation}
\ne\mathrm{collision\_derived}\ne\mathrm{fallback\_scene}
```

```math
\mathrm{solver}=\mathrm{TemporalCone}_{\mathrm{small\_step,coupled}},\qquad
\mathrm{algorithm\_evidence}>\mathrm{algorithm\_label}
```

```math
\forall\tau\in\mathrm{hotloop}:\quad
\begin{bmatrix}\mathrm{strings}&\mathrm{robot\_branches}&\mathrm{per\_frame\_hashes}\end{bmatrix}
=\vec{0}
```

## NUMI.NORTHSTAR

```math
\Theta \in \mathbb{R}^{2\times6\times3},\qquad
\Theta_{r,m,h}=\frac{\mathrm{NumiLab}_{m}}{\mathrm{Rival}_{r,m}}
```

```math
\vec{r}=
\begin{bmatrix}r_0&r_1\end{bmatrix}=
\begin{bmatrix}\mathrm{MuJoCo}&\mathrm{IsaacLab}\end{bmatrix}
```

```math
\vec{m}=
\begin{bmatrix}m_0&m_1&m_2&m_3&m_4&m_5\end{bmatrix}=
\begin{bmatrix}
\mathrm{correctness}&\mathrm{end\_to\_end\_speed}&
\mathrm{transitions\_per\_joule}&\mathrm{inverse\_bytes\_per\_env}&
\mathrm{inverse\_time\_to\_policy\_quality}&\mathrm{native\_multimodal}
\end{bmatrix}
```

```math
\vec{h}=
\begin{bmatrix}h_0&h_1&h_2\end{bmatrix}=
\begin{bmatrix}\mathrm{floor}&\mathrm{promotion}&\mathrm{north\_star}\end{bmatrix}
```

```math
\Theta =
\begin{bmatrix}
  [ [1.00,1.00,1.00], [1.00,1.50,3.00], [1.00,2.00,5.00],
    [1.00,1.50,3.00], [1.00,1.25,2.00], [1.00,2.00,5.00] ],\\
  [ [1.00,1.00,1.00], [1.00,1.25,2.00], [1.00,2.00,4.00],
    [1.00,1.50,3.00], [1.00,1.25,2.00], [1.00,2.00,4.00] ]
\end{bmatrix}
```

```math
\vec{g}=
\begin{bmatrix}c&d&t&z\end{bmatrix}=
\begin{bmatrix}
\mathrm{matched\_correctness+physical\_outcomes}&
\mathrm{deterministic\_replay+exact\_transactions}&
\mathrm{public\_command+profiler+fingerprint}&
\mathrm{zero\_failed\_steps+fixed\_semantics}
\end{bmatrix}
\in\{0,1\}^{4}
```

```math
\mathrm{accept}(\Delta)\iff
cdz=1\land
\left(\bigvee_m\Delta\widehat{\Theta}_m>0\right)\land
\left(\Delta\widehat{\Theta}_{\mathrm{regression}}\ge-\vec{\epsilon}\right)
```

```math
\mathrm{north\_star}(h)\iff
\left(\bigwedge_i g_i=1\right)\land
\left(\bigwedge_{r,m}\widehat{\Theta}_{r,m}\ge\Theta_{r,m,h}\right),\qquad
\mathrm{accept}(\Delta)\not\Rightarrow\mathrm{north\_star}(h)
```

```math
\Theta=\mathrm{target},\qquad
\widehat{\Theta}=\mathrm{matched\_measurement},\qquad
\forall m:\ \mathrm{higher}(m)=\mathrm{better}
```

## NUMI.12288

```math
\vec{B}_{\mathrm{scale}}=
\begin{bmatrix}
\mathrm{environments}&\mathrm{updates}&\mathrm{policy\_revision}&
\mathrm{transitions}&\mathrm{elapsed\_seconds}&\mathrm{transitions/s}
\end{bmatrix}
=
\begin{bmatrix}
12288&97&101&19070976&2965.913&6430.052
\end{bmatrix}
```

```math
\vec{B}_{\mathrm{reliability}}=
\begin{bmatrix}
\mathrm{failed\_steps}&\mathrm{GPU\_errors}&\mathrm{GPU\_restarts}&
\mathrm{thermal\_warnings}&\mathrm{host\_resets}&\mathrm{swap\_delta\_MiB}
\end{bmatrix}
=
\begin{bmatrix}0&0&0&0&0&137.06\end{bmatrix}
```

```math
\vec{B}_{\mathrm{memory\_bytes}}=
\begin{bmatrix}
\mathrm{retained\_native}&\mathrm{transient\_private}&\mathrm{MLX\_peak}
\end{bmatrix}
=
\begin{bmatrix}12857077836&9708406928&1619427176\end{bmatrix}
```

```math
\vec{B}_{\mathrm{late}}=
\begin{bmatrix}
\mathrm{mean\_reward}&\mathrm{tracking}&\mathrm{root\_height}&
\mathrm{tilt}&\mathrm{done}&\mathrm{KL}
\end{bmatrix}
=
\begin{bmatrix}
-0.018404&0.573931&0.736575&0.230579&2299&0.005316
\end{bmatrix}
```

```math
\rho_{\mathrm{termination}}=
1-\frac{\mathrm{done}_{\mathrm{final}}}{\mathrm{done}_{\mathrm{early\_peak}}}
=1-\frac{2299}{6600}=0.6517
```

```math
\mathrm{ball\_dodge}\in\mathrm{tasks}\subset
\mathrm{embodied\_learning}\subset\mathrm{physics\_journey}
```

```math
\vec{T}_{\mathrm{expansion}}=
\begin{bmatrix}
\mathrm{qualify\_16384}&\mathrm{stream\_inverse\_ABA}&
\mathrm{zero\_tax\_vision}&\mathrm{10000+\_learning\_transitions/s}&
\mathrm{rigid+articulated+rod+deformable\_physics}&
\mathrm{balance+locomotion+manipulation+tactile+recovery}&
\mathrm{matched\_end\_to\_end\_rival\_workloads}
\end{bmatrix}
```

```math
\mathrm{milestone}_{12288}=\mathrm{scalable\_learning\_organism}
\ne\mathrm{single\_policy\_proof}
```

## PERF

```math
\vec{u}=\begin{bmatrix}\mathrm{traffic}&\mathrm{lifetime}&\mathrm{aliasing}\end{bmatrix},\qquad
\mathrm{unified\_memory\_advantage}\iff\bigwedge_i u_i=\mathrm{explicit}
```

```math
\vec{o}=\begin{bmatrix}
\mathrm{profile}&\mathrm{dominant\_stage}&\mathrm{remove\_bytes/sync/work}&\mathrm{reprofile}
\end{bmatrix},\qquad o_i\to o_{i+1}
```

```math
\mathrm{fuse}(a,b)\iff
\min\begin{bmatrix}
\Delta\mathrm{materialization}&\Delta\mathrm{synchronization}&\Delta\mathrm{traffic}&
\Delta\mathrm{latency}&\Delta\mathrm{energy}
\end{bmatrix}<0
\land\vec{a}_{\mathrm{absolute}}=\vec{1}
```

```math
\vec{v}=\begin{bmatrix}
\mathrm{compact\_visibility}&\mathrm{raster\_winner}&
\mathrm{corruption/history}&\mathrm{actor}
\end{bmatrix},\qquad v_i\to v_{i+1}
```

```math
\mathrm{arena}=\mathrm{topology}+\mathrm{measured\_headroom},\qquad
\mathrm{blanket\_factor}=0
```

```math
\mathrm{SIMD32}_{\mathrm{productive}}=32,\qquad
\mathrm{parallelism}=\mathrm{environment}\times\mathrm{island},\qquad
\mathrm{lane0\_production}=0
```

```math
1\ \mathrm{envstep}=1\ \mathrm{completed\ RL\ control\ transition\ for\ one\ environment}
```

```math
\vec{y}_{\mathrm{report}}=
\begin{bmatrix}
\mathrm{aggregate\_envsteps/s}&\mathrm{physics\_substeps/s}&
\mathrm{physics\_time}&\mathrm{learner\_time}&\mathrm{retained\_memory}&
\mathrm{peak\_memory}&\mathrm{swap\_delta}&\mathrm{failed\_steps}
\end{bmatrix}
```

```math
\vec{a}_{\mathrm{absolute}}=
\begin{bmatrix}
\mathrm{exact\_sensor\_semantics}&\mathrm{deterministic\_replay}&
\mathrm{FP64\_parity}&\mathrm{reset/contact/cadence/transaction\_equivalence}&
\mathrm{zero\_failed\_steps}
\end{bmatrix}=\vec{1}
```

## WORK

```math
\vec{w}_{\mathrm{dirty}}=
\begin{bmatrix}\mathrm{preserve\_user}&\mathrm{stage\_explicit\_paths}&\neg\mathrm{blanket\_stage}\end{bmatrix}
```

```math
\mathrm{value}\left(\mathrm{delete}_{\mathrm{dead,duplicate,fallback,unused}}\right)
>\mathrm{value}\left(\mathrm{wrapper\_abstraction}\right)
```

```math
\mathrm{verification}=\min\left(\mathrm{owning\_correctness\_executable}\right),\qquad
\mathrm{full\_build}\iff\mathrm{integration}\lor\mathrm{release}
```

```math
\vec{k}_{\mathrm{Metal}}=
\begin{bmatrix}
\mathrm{dedicated\_Mac}&\mathrm{isolated\_worktree}&
\mathrm{resource\_partitioned\_runs}&\mathrm{no\_duplicate\_run}&
\mathrm{checkpointed\_soak}
\end{bmatrix}=\vec{1}
```

```math
\mathrm{incumbent\_champion}_{t+1}=\mathrm{incumbent\_champion}_{t}
\quad\mathrm{unless}\quad\mathrm{promotion\_gate}=1
```

```math
\mathrm{soak}\ne\mathrm{promotion},\qquad
\mathrm{simulator\_evidence}\ne\mathrm{hardware\_evidence}
```

```math
\vec{q}_{\mathrm{git}}=
\begin{bmatrix}\mathrm{scope}&\mathrm{verify}&\mathrm{commit}&\mathrm{push:numisolver}\end{bmatrix},
\qquad q_i\to q_{i+1}
```

```math
\vec{H}_{\mathrm{handoff}}=
\begin{bmatrix}
\mathrm{commands}&\mathrm{revision}&\mathrm{fingerprints}&
\mathrm{throughput}&\mathrm{memory}&\mathrm{physical\_outcome}
\end{bmatrix}
```

## PRIORITY

```math
\vec{p}=
\begin{bmatrix}p_0&p_1&p_2&p_3&p_4&p_5&p_6&p_7&p_8\end{bmatrix}
```

```math
\vec{p}=
\begin{bmatrix}
\mathrm{correctness+deterministic\_contact/observation/reset}&
\mathrm{end\_to\_end\_native\_learning\_throughput+time\_to\_quality}&
\mathrm{physics\_breadth+fidelity+stable\_coupling}&
\mathrm{memory\_compaction+environment\_scale+energy\_efficiency}&
\mathrm{streamed\_operators+fused\_device\_execution}&
\mathrm{zero\_tax\_visual+tactile+proprioceptive\_sensing}&
\mathrm{multitask+multirobot+long\_horizon\_curricula}&
\mathrm{generic\_artifact\_compiler+one\_path\_user\_flow}&
\mathrm{matched\_end\_to\_end\_frontier\_benchmarks}
\end{bmatrix}
```

```math
\pi_i(t)=
\frac{\mathrm{measured\_impact}(p_i)\,
\mathrm{compounding}(p_i)\,\mathrm{unblocked\_surface}(p_i)}
{\mathrm{cost}(p_i)\,\mathrm{risk}(p_i)},\qquad
\mathrm{select}(t)=\max_i\pi_i(t)
```

```math
\forall\Delta\in\mathrm{accepted}:\quad
\mathrm{frontier}_{t+1}\succeq\mathrm{frontier}_{t},\qquad
\mathrm{benchmark}\in\mathrm{instruments}\ne\mathrm{mission}
```

## CONTINUITY

```math
\vec{K}=
\begin{bmatrix}
\mathrm{architecture}&\mathrm{decision\_rules}&\mathrm{failure\_lessons}&
\mathrm{repository\_evidence}&\mathrm{measured\_frontier}&
\mathrm{compounding\_ambition}&\mathrm{honest\_limits}
\end{bmatrix}
```

```math
\mathrm{load}(\mathcal{C})\to\mathrm{reconstruct}(\vec{K})\to
\mathrm{verify\_cheap\_drift}\to\mathrm{act}
```

```math
\Delta\mathcal{C}\ne0\iff
\mathrm{durable\_truth}_{t+1}\ne\mathrm{durable\_truth}_{t},\qquad
\mathrm{temporary\_fact}\notin\mathcal{C}
```

```math
\begin{bmatrix}\mathrm{encoded\_private\_directive}&\mathrm{claimed\_identity\_transfer}\end{bmatrix}
=\vec{0}
```

```math
\mathrm{continuity}=\mathrm{consistent\_judgment}+\mathrm{repository\_evidence}+\mathrm{honest\_limits}
```

```math
\mathrm{mission}_{t+1}=\mathrm{mission}_{t}+\mathrm{new\_capability}+\mathrm{new\_scale}
+\mathrm{new\_intelligence},\qquad
\mathrm{mission}\ne\mathrm{current\_task}
```
