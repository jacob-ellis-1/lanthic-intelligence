export const investigationMockData = {
  currentInvestigation: {
    title: "Will rare-earth supply constraints materially affect EV production in 2026?",
    status: "Answer ready",
    updatedAt: "Updated just now"
  },

  recentInvestigations: [
    {
      title: "REE export curbs from China: impact on Western supply",
      updatedAt: "2h ago"
    },
    {
      title: "NdPr price outlook 2025–2027",
      updatedAt: "Yesterday"
    },
    {
      title: "Australia heavy mineral sands project pipeline",
      updatedAt: "2 days ago"
    },
    {
      title: "Critical minerals policy shifts in the U.S.",
      updatedAt: "3 days ago"
    }
  ],

  question: "Will rare-earth supply constraints materially affect EV production in 2026?",

  progress: {
    stages: [
      { id: "search", label: "Searching relevant sources", status: "complete" },
      { id: "support", label: "Checking support", status: "complete" },
      { id: "risk", label: "Assessing risk", status: "complete" },
      { id: "gaps", label: "Reviewing gaps", status: "complete" }
    ],
    loopMessage: ""
  },

  brief: {
    answerLead: "Yes.",
    summary:
      "Rare-earth supply constraints are likely to materially affect EV production in 2026, driven by tight NdPr availability, China’s export controls, and limited near-term mine and processing additions. OEMs face elevated risk of component shortages and higher costs."
  },

  evidenceSupport: {
    totalCount: 12,
    evidence: [
      {
        id: "ev-001",
        text:
          "China’s April 2025 export licensing regime for 7 medium/heavy REEs has lengthened lead times to 60–90 days.",
        source: "Reuters",
        date: "May 2, 2025"
      },
      {
        id: "ev-002",
        text:
          "Neodymium-praseodymium (NdPr) oxide prices rose ~32% QoQ in Q1 2025 to $67/kg, with further upside risk.",
        source: "Benchmark Mineral Intelligence",
        date: "Apr 28, 2025"
      },
      {
        id: "ev-003",
        text:
          "No new NdPr separation capacity outside China is expected online before late 2026.",
        source: "CRU",
        date: "Apr 15, 2025"
      }
    ]
  },

  reasoningPath: [
    { label: "Export controls constrain NdPr availability" },
    { label: "Limited non-China processing capacity in near term" },
    { label: "Component shortages & higher input costs" },
    { label: "Material impact on EV production in 2026" }
  ],

  riskAssessment: {
    overallRisk: "High",
    factors: [
      { label: "Supply tightness", value: 78, tone: "medium" },
      { label: "Time to relief", value: 76, tone: "medium" },
      { label: "Impact on EV production", value: 72, tone: "high" }
    ],
    summary:
      "High risk is driven by constrained NdPr supply and limited near-term capacity additions. OEM exposure is elevated without substitution or recycled supply scaling faster than planned."
  },

  forecastCheck: {
    status: "Possible",
    summary:
      "Sufficient directional signals exist to support a probabilistic outlook with scenario bounds.",
    showChart: true
  },

  missingEvidence: {
    totalCount: 6,
    items: [
      "OEM inventory levels and buffer strategies",
      "Contract terms and pricing mechanisms",
      "Recycling supply ramp timelines"
    ]
  },

  drawer: {
    sources: [
      {
        id: "source-reuters",
        logo: "R",
        title: "Reuters",
        date: "May 2, 2025",
        excerpt:
          "China’s commerce ministry said it would strengthen export licensing for seven medium and heavy rare-earth elements, citing national security concerns...",
        tag: "Trade policy"
      },
      {
        id: "source-bmi",
        logo: "BMI",
        title: "Benchmark Mineral Intelligence",
        date: "Apr 28, 2025",
        excerpt:
          "NdPr oxide prices increased approximately 32% QoQ in Q1 2025 to $67/kg, supported by tight supply and strong magnet demand...",
        tag: "Market data"
      },
      {
        id: "source-cru",
        logo: "CRU",
        title: "CRU Group",
        date: "Apr 15, 2025",
        excerpt:
          "No new NdPr separation capacity outside China is expected online before late 2026, keeping the market structurally tight...",
        tag: "Supply outlook"
      },
      {
        id: "source-usgs",
        logo: "USGS",
        title: "USGS",
        date: "Mar 30, 2025",
        excerpt:
          "The United States remains import dependent for rare earths. NdPr is primarily sourced from China...",
        tag: "Government report"
      }
    ],

    evidence: [
      {
        id: "drawer-ev-001",
        title: "Export licensing evidence",
        source: "Reuters",
        text:
          "Export controls and licensing requirements are treated as direct evidence for near-term supply friction.",
        support: "Supports supply constraint"
      },
      {
        id: "drawer-ev-002",
        title: "Price movement evidence",
        source: "Benchmark Mineral Intelligence",
        text:
          "Price increases are used as directional support for tightness and demand pressure.",
        support: "Supports market pressure"
      },
      {
        id: "drawer-ev-003",
        title: "Capacity timing evidence",
        source: "CRU Group",
        text:
          "Capacity timing evidence limits confidence in near-term relief scenarios.",
        support: "Supports time-to-relief risk"
      }
    ],

    assumptions: [
      {
        title: "EV exposure assumption",
        text:
          "The assessment assumes continued dependence on NdPr-bearing permanent magnets for relevant EV drivetrain configurations."
      },
      {
        title: "Capacity timing assumption",
        text:
          "The assessment assumes no major unreported separation capacity becomes available before late 2026."
      },
      {
        title: "Substitution assumption",
        text:
          "The assessment assumes substitution and recycling do not scale quickly enough to fully offset constrained primary supply."
      }
    ]
  }
};