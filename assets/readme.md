## Semantic Planning Visualization

SPARK uses the MLLM as an offline **semantic planner** rather than a costly test-time judge. During semantic planning, high-confidence normal and abnormal event prototypes are summarized into class-specific keyword priors, which are then fed back to guide the next round of event generation.

This keyword evolution process progressively sharpens the semantic event space. Normal keywords gradually evolve toward **smooth and orderly traffic-flow semantics**, while abnormal keywords become increasingly associated with **risk, obstruction, collision, and accident-related events**. This indicates that SPARK does not rely on fixed hand-crafted prompts, but instead builds a scene-conditioned semantic prior through iterative planning.


### What does the evolution show?

| Class | Epoch 2 Semantics | Newly Refined Semantics in Epoch 3 | Interpretation |
|---|---|---|---|
| **Normal** | `driving`, `highway`, `vehicles`, `traffic`, `pedestrians`, `lanes` | `smooth`, `flowing`, `traffic lights`, `orderly` | Normal events become more associated with smooth traffic flow, safe navigation, and orderly scene dynamics. |
| **Abnormal** | `debris`, `erratic`, `swerve`, `collision`, `obstruction` | `obstruct`, `narrowly`, `deflect`, `hazards`, `accident`, `missteps` | Abnormal events become more risk-aware and accident-oriented, capturing stronger anomaly-related semantics. |

This evolution shows that SPARK does not rely on fixed hand-crafted prompts.  
Instead, it progressively constructs a **scene-conditioned semantic event space**, enabling the frozen VLM executor to perform efficient posterior inference over normal and abnormal event prototypes.

<details>
<summary><strong>Full keyword lists from semantic planning</strong></summary>

<br>

#### Epoch 2

**Normal Keywords**

```text
driving, highway, night, vehicles, stopped, traffic, lights, pedestrians,
cross, safely, smoothly, traveling, multi-lane, group, pass, busy,
standing, sidewalk, intersection, control, flow, navigate, lanes,
mostly, clear, visible
```

**Abnormal Keywords**

```text
debris, erratic, vehicle, swing, across, sudden, lane, collision,
middle, road, drive, shoulder, signaled, swerve, narrow, obstruction
```

#### Epoch 3

**Normal Keywords**

```text
driving, highway, night, vehicles, stopped, traffic, lights, pedestrians,
cross, safely, smoothly, traveling, multi-lane, group, pass, busy,
standing, sidewalk, intersection, control, flow, navigate, lanes,
mostly, clear, visible, smooth, flowing, traffic lights, orderly
```

**Abnormal Keywords**

```text
debris, erratic, vehicle, swing, across, sudden, lane, collision,
middle, road, drive, shoulder, signaled, swerve, narrow, obstruction,
obstruct, narrowly, deflect, sways, hazards, accident, missteps
```

</details>
