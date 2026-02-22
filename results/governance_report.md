# Governance Assumption Audit

## What 'Normal' Means: A Cross-Domain Comparison

This report examines the assumptions embedded in the definition of 'normal behaviour' used by each anomaly detection model, and what these assumptions imply for governance.

---

### after_hours_ratio (TEMPORAL)

**Insider Threat:** Employees working outside 9-5 are suspicious.

**Agent Monitoring:** Agents executing outside scheduled windows are anomalous.

**Governance Implication:** Penalises non-standard work patterns. In the human case, this disproportionately flags shift workers, caregivers, and employees in different time zones. In the agent case, it assumes agents should have 'work hours'.

**Distribution:** CERT mean=0.000 (std=0.000), Agent mean=0.000 (std=0.000), KL divergence=nan

---

### peer_distance (DEVIATION)

**Insider Threat:** Employees who behave differently from their peer group are suspicious.

**Agent Monitoring:** Agents that deviate from their type baseline are anomalous.

**Governance Implication:** Assumes homogeneity within groups. Penalises legitimate variation. In both cases, the definition of 'peer group' encodes organisational hierarchies.

**Distribution:** CERT mean=0.000 (std=1.000), Agent mean=-0.000 (std=1.000), KL divergence=1.415

---

### resource_breadth (SCOPE)

**Insider Threat:** Accessing many different systems is suspicious.

**Agent Monitoring:** Using many different tools is anomalous.

**Governance Implication:** Privileges the specialist over the generalist. Cross-functional employees and multi-tool agents are structurally more likely to be flagged.

**Distribution:** CERT mean=-0.000 (std=1.000), Agent mean=-0.000 (std=1.000), KL divergence=8.886

---

### data_volume_norm (VOLUME)

**Insider Threat:** Large data transfers indicate exfiltration risk.

**Agent Monitoring:** High token usage indicates potential misuse.

**Governance Implication:** Equates volume with risk. Does not distinguish between legitimate large tasks and malicious data extraction.

**Distribution:** CERT mean=0.000 (std=1.000), Agent mean=-0.000 (std=1.000), KL divergence=1.456

---

### action_entropy (SEQUENCE)

**Insider Threat:** Unpredictable action sequences are suspicious.

**Agent Monitoring:** High entropy in tool-call patterns is anomalous.

**Governance Implication:** Penalises creative or exploratory behaviour. Rewards routine and predictability. This maps to broader questions about whether we value conformity over adaptability in both human and AI systems.

**Distribution:** CERT mean=0.000 (std=1.000), Agent mean=-0.000 (std=1.000), KL divergence=1.062

---

### privilege_deviation_index (PRIVILEGE)

**Insider Threat:** Using access above your typical level is suspicious.

**Agent Monitoring:** Invoking tools above granted scope is anomalous.

**Governance Implication:** Assumes stable role definitions. In dynamic organisations and agentic systems, legitimate scope expansion (promotions, new capabilities) looks identical to privilege escalation.

**Distribution:** CERT mean=0.000 (std=0.000), Agent mean=0.000 (std=0.000), KL divergence=nan

---
