# State of the Art in Automated Privacy Policy Analysis and GDPR Compliance Assessment

*Literature and market review — prepared to ground the "differentiation from SOTA" section of a white paper on a RAG + fine-tuned open-source LLM judge for GDPR privacy-policy scoring.*

*Compiled: August 2026*

---

## 1. Academic approaches to automated privacy policy analysis

### 1.1 Polisis (2018)

**Method.** Polisis (Harkous et al., USENIX Security 2018) is a deep-learning pipeline built on hierarchical, category-specific CNN/RNN classifiers. It segments a privacy policy into text blocks and predicts, per segment, a data practice category (e.g., first-party collection, third-party sharing, retention, choice/control) plus fine-grained attributes (data type, purpose, choice type). It powers **PriBot**, a conversational agent that answers free-text questions about a given policy by retrieving the most relevant classified segment. This makes Polisis the closest historical ancestor of a "chat with a policy" interface, though it predates GDPR-specific framing and predates the retrieval-augmented-generation paradigm — its "retrieval" is a similarity search over its own classifier outputs, not a modern RAG stack over regulatory text.

**Dataset.** Trained/evaluated on the **OPP-115** corpus (115 policies, ~23,000 practice annotations) for the categorization task, with additional crowdsourced data for the question-answering (PriBot) evaluation.

**Granularity.** Segment/paragraph-level classification into ~10 high-level categories with sub-attributes; not GDPR-article-level and not a compliance score.

**Public availability.** Partial. The original authors did not release a full production repo, but independent reimplementations of the Polisis classifiers exist on GitHub (e.g., `quanmou/polisis`, `Maxikilliane/polisis-classifiers`), and a hosted demo was available via `pribot.org`. No official fine-tuned model weights were released.

Sources: [Polisis (USENIX Security '18)](https://www.usenix.org/conference/usenixsecurity18/presentation/harkous), [arXiv:1802.02561](https://arxiv.org/abs/1802.02561), [pribot.org](https://pribot.org/), [quanmou/polisis](https://github.com/quanmou/polisis)

### 1.2 PolicyLint (2019)

**Method.** PolicyLint (Andow et al., USENIX Security 2019) is a classical NLP pipeline (ontology generation, dependency parsing, named-entity recognition, pattern matching) that extracts privacy statements as four-tuples — *(actor, action, data object, entity)* — and then runs logical contradiction analysis across tuples within a single policy. It is not a compliance scorer; it is a **contradiction/consistency checker**, flagging cases where a policy simultaneously claims to collect and not-collect the same data type, or contradicts itself about sharing with third parties.

**Dataset.** 11,430 Android app privacy policies crawled from the top 500 free apps in each of Google Play's 35 categories (Sept. 2017).

**Granularity.** Fine-grained, tuple-level extraction; findings categorized into 9 contradiction types.

**Public availability.** No public code or model release could be confirmed from the paper or supplementary materials; it is a research prototype rather than an open toolkit.

Sources: [PolicyLint (USENIX Security '19)](https://www.usenix.org/conference/usenixsecurity19/presentation/andow), [PDF](https://www.usenix.org/system/files/sec19-andow.pdf)

### 1.3 CLAUDETTE (2018–2019)

**Method.** CLAUDETTE (Lippi, Contissa, Lagioia, Micklitz, Pałka, Sartor, Torroni) applies supervised machine learning (SVM/CNN-based sentence classifiers, later extended) to Terms of Service and, in the "CLAUDETTE meets GDPR" extension, to privacy policies specifically, to flag **potentially unfair or non-compliant clauses**. The GDPR extension maps sentences to specific fairness/transparency issues drawn from GDPR provisions (e.g., missing legal basis, vague retention periods, unclear data-subject rights language). It is the closest pre-LLM academic system to "compliance judging" rather than pure information extraction, and it is grounded in EU consumer/data-protection law rather than being law-agnostic like OPP-115-based work.

**Dataset.** Terms of service from 47 major online platforms (training) plus a 10-platform validation set for the base CLAUDETTE work; the GDPR extension uses a curated set of privacy policies annotated by legal experts against GDPR fairness/transparency criteria.

**Granularity.** Sentence-level binary/multiclass classification (fair / potentially unfair / clearly unfair), with clause-type labels.

**Public availability.** The ToS training corpus is public (`claudette.eui.eu/ToS.zip`), and a live web demo/tool is hosted at `claudette.eui.eu`. Full model weights and training code were not confirmed as openly released; the project is best characterized as "data public, service public, training pipeline not fully open."

Sources: [CLAUDETTE — AI & Law, Springer](https://link.springer.com/article/10.1007/s10506-019-09243-2), [CLAUDETTE meets GDPR](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3208596), [claudette.eui.eu](http://claudette.eui.eu/)

### 1.4 PolicyGPT (2023)

**Method.** PolicyGPT (arXiv:2309.10238) is a **zero-shot LLM prompting framework** (using ChatGPT/GPT-4 as off-the-shelf classifiers, no fine-tuning) for privacy-policy segment classification. It represents the pivot point in the literature from trained classical/deep classifiers to prompted general-purpose LLMs, and it is frequently cited as the direct predecessor of the "LLM as privacy-policy analyst" line of work this project extends by adding RAG and a compliance-judgment layer.

**Dataset.** Two corpora: (1) 115 website policies with legal-expert annotations across 10 classes (essentially OPP-115-style), and (2) 304 mobile app policies with sentence-level annotations across 10 categories.

**Granularity.** Segment/sentence-level, 10-class data-practice categorization (not GDPR-article-level, not a pass/fail or numeric compliance score).

**Public availability.** No confirmed public code/model repository; the method depends on closed commercial APIs (OpenAI), so there is no "model" to release in the open-weight sense.

Sources: [PolicyGPT (arXiv:2309.10238)](https://arxiv.org/abs/2309.10238), [HF paper page](https://huggingface.co/papers/2309.10238)

### 1.5 2024–2026 work using LLMs as GDPR compliance judges

Three recent works are directly relevant as near-neighbors to this project and should be discussed explicitly in the differentiation section:

**(a) Word-level GDPR Transparency Annotation with LLMs (arXiv:2503.10727, 2025).** A modular LLM pipeline combining passage-level classification, **retrieval-augmented generation**, and a self-correction step to produce **word/span-level** annotations against **21 GDPR-derived transparency requirements**. Evaluated on a purpose-built ground-truth set of 200 manually annotated policies plus comparison on OPP-115, over a base corpus of 703,791 policies. This is the closest prior art to the proposed system's RAG design, but it targets *transparency* requirements specifically (Art. 13/14-style disclosure obligations) rather than a broader GDPR compliance score, and public code/model release was not confirmed from the abstract-level content reviewed. A fine-tuned open-weight judge model, if released, would be a meaningful open-source differentiator versus this closed/proprietary-LLM pipeline.

**(b) "Evaluating Privacy Policies under Modern Privacy Laws at Scale: An LLM-Based Automated Approach" (Xie, Ramakrishnan et al., USENIX Security 2025).** Applies an LLM-based method to privacy policies from over 100,000 websites, systematizing **10 privacy regulations** (GDPR and multiple US state laws) into **34 distinct clauses across 4 themes**, and using the LLM to judge whether each clause is addressed. This is the most directly comparable "LLM-as-judge for policy compliance at scale" system found in the 2024–2026 window. Notably, it carries a USENIX **Artifact Evaluation Available** badge, suggesting partial code/data release — this should be verified directly against the artifact appendix before claiming novelty over it. It is multi-jurisdictional (not GDPR-specific) and, per available materials, does not appear to use a fine-tuned open-source judge model — it relies on prompting large proprietary models.

**(c) LLM-Driven GDPR Compliance Checking for Android Apps (FSE 2025, "RegCheck").** Extends the LLM-as-compliance-checker paradigm to the code/manifest layer of Android apps rather than policy text alone, cross-referencing declared behavior against GDPR obligations — useful as an adjacent-but-distinct line of work (code-behavior vs. policy-text compliance) worth citing to bound the scope of "privacy policy text analysis" claimed by this project.

Other adjacent 2024–2025 items worth a citation-list mention: LLM-assisted extraction of regulatory requirements from GDPR text itself (Orbilu/Luxembourg, RE'25) — relevant as a technique for building the RAG knowledge base of GDPR provisions rather than for scoring policies; and general "LLM privacy policy analysis at scale" work (Computing, Springer 2024, building on PolicyGPT-style prompting) confirming the field's rapid move toward prompted-LLM pipelines without, so far, a widely cited **fine-tuned, open-weight** GDPR judge model.

Sources: [Word-level GDPR annotation (arXiv:2503.10727)](https://arxiv.org/abs/2503.10727), [Evaluating Privacy Policies at Scale (USENIX Sec '25)](https://www.usenix.org/conference/usenixsecurity25/presentation/xie), [RegCheck (FSE 2025 preprint)](https://jacquesklein2302.github.io/papers/2025-FSE-IVR-RegCheck.pdf), [LLM-assisted GDPR requirement extraction](https://orbilu.uni.lu/bitstream/10993/65265/1/2025-RE-ACSBLSVS.pdf), [Large Language Models: a new approach for privacy policy analysis at scale (Computing, 2024)](https://link.springer.com/article/10.1007/s00607-024-01331-9)

**Caveat on this subsection:** several of the 2024–2026 sources above were reviewed via abstracts, HTML previews, or third-party summaries rather than full PDF close-reads (some full texts sit behind paywalls or returned partial extracts). Before finalizing white-paper claims of novelty, the full text of (a) and (b) in particular should be read directly to confirm code/model release status and to check they do not already ship a fine-tuned open judge model.

---

## 2. Commercial compliance tools

The five vendors requested — OneTrust, TrustArc, Osano, Termly, iubenda — split into two functional tiers that matter for differentiation framing:

- **Enterprise privacy management platforms** (OneTrust, TrustArc): primarily *workflow and data-mapping* products (DPIAs/PIAs, consent management, vendor risk assessment, records of processing) into which "AI" features have recently been layered for document summarization and assessment auto-population — not independent, from-scratch compliance scoring of a policy's *text*.
- **SMB-focused policy generation / consent tooling** (Osano, Termly, iubenda): primarily *policy generators and cookie-consent banner managers* aimed at helping a site *produce* a compliant policy from a questionnaire, with some post-hoc scanning of the live site (banner presence, consent signal handling) rather than deep textual compliance analysis of the policy's substantive claims.

**OneTrust.** Markets "AI Document Scanning" and "AI Inventory Analysis" that auto-populate compliance assessments from uploaded documentation (contracts, security certifications, existing data maps) and a "Copilot" that surfaces regulatory guidance from a licensed dataset (DataGuidance, curated by in-house privacy experts). Public marketing materials state general responsible-AI guardrails ("models are not trained on your data," human-in-the-loop review) but do **not** disclose the scoring rubric, model architecture, validation methodology, or how (or whether) GDPR articles map to specific automated checks.

**TrustArc.** Offers "Intelligent GDPR Assessments" and an Assessment Manager/PrivacyCentral product line for structured, questionnaire-driven privacy and vendor risk assessments, plus a certification/validation service. The certification process involves TrustArc's own (human) assessors: it is positioned as compliance *program management and human-backed certification*, not an automated NLP judge of policy text.

**Osano.** "Compliance Check" performs automated scanning of a live website for consent-banner presence, dark-pattern flags, Global Privacy Control (GPC)/IAB string/Google Consent Mode handling, and basic policy metadata (presence, last-updated date, accessibility) — this is close to the proposed project's use case in spirit (automated, low-touch scanning) but appears to operate primarily at the **technical/consent-signal layer**, not deep semantic analysis of policy clause content against GDPR articles.

**Termly and iubenda.** Both are primarily **policy generation** tools: a business answers a structured questionnaire and receives a templated privacy/cookie policy assembled from a library of pre-approved clauses mapped to selected jurisdictions (GDPR, CCPA, etc.). Assessment, where offered, is oriented around confirming the generated document matches the selected template requirements — not independent NLP/LLM analysis of an arbitrary, already-published policy.

**Limitation of this section, stated explicitly:** for all five vendors, the underlying scoring logic, model (if any), validation data, and accuracy benchmarks are **proprietary and not publicly documented**. Everything above is derived from marketing pages, product documentation, and press releases; none of it constitutes a peer-reviewed or independently audited account of methodology. This opacity is itself a legitimate, citable gap: none of the commercial incumbents publish a transparent, reproducible scoring methodology, an explicit mapping from output to specific GDPR articles, or open validation data — which is precisely the kind of claim a research project with a documented RAG pipeline and evaluation protocol can credibly contrast itself against.

Sources: [OneTrust AI privacy automation](https://www.onetrust.com/scale-privacy-with-ai/), [OneTrust privacy automation](https://www.onetrust.com/solutions/privacy-automation/), [TrustArc Intelligent GDPR Assessments](https://www.prnewswire.com/news-releases/trustarc-launches-intelligent-gdpr-assessments-to-drive-compliance-automation-300612598.html), [TrustArc GDPR validation](https://trustarc.com/products/assurance-certifications/gdpr-validation/), [TrustArc Assessment Manager](https://trustarc.com/products/privacy-data-governance/assessment-manager/), [Osano Compliance Check](https://www.osano.com/features/compliance-check), [Osano Compliance Check announcement](https://www.osano.com/updates/osano-compliance-check-automated-proof-website-privacy-compliance), [Termly](https://termly.io/), [Termly — laws covered](https://termly.io/resources/articles/which-laws-does-termly-cover/), [iubenda GDPR compliance](https://www.iubenda.com/en/help/140769-gdpr-compliance-made-easy/), [iubenda privacy policy generator](https://www.iubenda.com/en/privacy-and-cookie-policy-generator/)

---

## 3. OPP-115 and complementary datasets

### 3.1 OPP-115 (Wilson et al., ACL 2016)

The foundational corpus for privacy-policy NLP: **115 website privacy policies**, annotated by 10 law students, yielding **~2,831–23,000 annotations** (figures vary by counting unit — practice-level vs. fine-grained attribute annotations — across citing papers) across **10 mutually exclusive high-level categories** (First Party Collection/Use, Third Party Sharing/Collection, Data Retention, Data Security, User Choice/Control, User Access/Edit/Deletion, Policy Change, International/Specific Audiences, Do Not Track, Other). It predates GDPR (2016, GDPR enforcement began 2018) and its category schema was designed to be **law-agnostic** — descriptive of how policies are typically structured, not derived from any specific regulation's article structure.

A dedicated mapping study ("Mapping the GDPR to a Privacy Policy Corpus Annotation Scheme") subsequently cross-walked OPP-115's 10 categories against GDPR (including all 99 articles) and found a structural mismatch: OPP-115 separates "First Party Collection/Use" from "Third Party Sharing/Collection" as distinct categories, whereas GDPR's Article 5 principles apply uniformly to all processing regardless of party. This is the clearest documented evidence that **OPP-115, used as-is, is not a GDPR-aligned annotation schema** — it requires a translation/mapping layer (which is itself a nontrivial research contribution) before it can ground GDPR-specific compliance judgments.

### 3.2 APP-350 (Zimmeck et al. / MAPS, PoPETs 2019)

Companion corpus of **350 mobile app privacy policies** (247 apps with 50M+ installs, plus 103 randomly sampled apps with 5M+ installs from Google Play), annotated by legal experts across three dimensions: **data type** (location, contacts, device identifiers, SSO, etc.), **party** (first vs. third), and **modality** (practice stated as performed vs. explicitly not performed). Annotation reliability reported at Krippendorff's α = 0.78. Publicly available via `data.usableprivacy.org`. Like OPP-115, it is descriptive/law-agnostic rather than GDPR-article-mapped, and its mobile-app focus (paired with the Google Play ecosystem and US-style disclosure norms) limits direct transfer to GDPR's broader material scope (any controller/processor, not just apps).

### 3.3 PrivaSeer (Srinath, Wilson, Giles, ACL 2021)

A **very large-scale, unannotated** corpus — over **one million English-language website privacy policies** — built via crawling, language filtering, deduplication, and content extraction, with associated tooling for readability scoring, similarity analysis, keyphrase extraction, and topic modeling. Its value for this project is as a **pretraining/RAG-corpus source and out-of-distribution evaluation set**, not as labeled ground truth: it carries no compliance or category annotations and is not GDPR-specific.

### 3.4 PolicyIE (Ahmad et al., ACL 2021)

A **smaller, richly annotated** corpus of 31 policies (25 websites, 16 mobile apps) framed as an **intent classification + slot filling** task (5,250 intent annotations, 11,788 slot annotations across 4,209 train / 1,041 test sentences). Five intent categories (Data Collection/Usage, Data Sharing/Disclosure, Data Storage/Retention, Data Security/Protection, Other) were deliberately chosen to align with the **four primary data-handling practices GDPR addresses**, with an 18-label slot schema (participants — Data Provider/Collector/Receiver — plus attributes like Purpose, Condition, Polarity, Protection Method) enabling structured, sentence-level extraction closer to "what does this clause actually say" than OPP-115's coarser categorical labels. Code and data are openly released on GitHub. This is the most GDPR-conscious of the classic annotated corpora, but its scale (31 policies) is far too small to fine-tune a modern LLM judge on its own, and it still stops short of an explicit article-by-article or compliance-pass/fail schema.

### 3.5 Summary of GDPR-specific limitations across these datasets

None of OPP-115, APP-350, PrivaSeer, or PolicyIE were built with an explicit, article-level GDPR compliance annotation schema (pass/fail or graded compliance per Article 5, 6, 12–14, 15–22, etc.). OPP-115 and APP-350 predate or are contemporaneous with early GDPR enforcement and reflect US-centric, CCPA/COPPA-adjacent disclosure norms (the same limitation the user's brief anticipated); PolicyIE gestures toward GDPR's four core practices but at a scale too small for LLM fine-tuning; PrivaSeer offers scale but no labels at all. This is a direct, evidence-backed gap this project's RAG-over-GDPR-text-plus-judge approach can claim to address, provided the project constructs (or the white paper explicitly flags as future work) a GDPR-article-aligned gold evaluation set — which appears not to exist publicly at meaningful scale as of this review.

Sources: [OPP-115 corpus](https://usableprivacy.org/data), [Mapping GDPR to OPP-115 annotation scheme](https://par.nsf.gov/servlets/purl/10257054), [MAPS / APP-350 (PoPETs 2019)](https://petsymposium.org/2019/files/papers/issue3/popets-2019-0037.pdf), [PrivaSeer (arXiv:2004.11131)](https://arxiv.org/abs/2004.11131), [PrivaSeer (ACL Anthology)](https://aclanthology.org/2021.acl-long.532/), [PolicyIE (arXiv:2101.00123)](https://arxiv.org/abs/2101.00123), [PolicyIE GitHub](https://github.com/wasiahmad/PolicyIE)

---

## 4. Comparison table

| Approach | Type | Transparency of scoring | Output granularity | Open-source (code/model/data) | LLM vs. classical NLP | RAG usage | GDPR-specific |
|---|---|---|---|---|---|---|---|
| **Polisis** (2018) | Academic | Method published; no compliance score, just categorization | Segment-level, ~10 categories | Data: OPP-115 (public). Code: unofficial reimplementations only. No official model release | Deep learning (CNN/RNN), pre-LLM | No (similarity search over classifier output, not modern RAG) | No — law-agnostic |
| **PolicyLint** (2019) | Academic | Method published; contradiction flags, not a score | Tuple-level (actor, action, data, entity); 9 contradiction types | Data: not released. Code: not confirmed public | Classical NLP (parsing, NER, ontology) | No | No |
| **CLAUDETTE** (2018–19) | Academic | Method published; sentence-level fair/unfair labels | Sentence-level, clause-type labels | Data: public (ToS.zip). Live web tool public. Full training code: not confirmed | Classical ML (SVM/CNN) | No | Partial — GDPR extension exists but is fairness-clause-focused, not full-article coverage |
| **PolicyGPT** (2023) | Academic | Method published; category labels, no numeric score | Segment/sentence-level, 10 categories | No public code/model (depends on closed OpenAI API) | LLM (zero-shot prompting, no fine-tuning) | No | No — general categorization |
| **Word-level GDPR Transparency Annotation** (arXiv:2503.10727, 2025) | Academic | Method published in detail; requirement-level pass/fail-style annotation | Word/span-level against 21 GDPR transparency requirements | Not confirmed from available materials | LLM (annotator + classifier) | **Yes** — explicit RAG component | **Yes** — but transparency (Art. 13/14) only, not full GDPR |
| **LLM Policy Evaluation at Scale** (Xie et al., USENIX Sec '25) | Academic | Method published; clause-level judgment across 34 clauses/4 themes | Clause-level, 10 laws incl. GDPR | Partial — carries "Artifact Evaluation Available" badge; exact scope unverified | LLM (prompted, likely proprietary models) | Not confirmed from abstract-level review | Multi-law, GDPR is one of 10 — not GDPR-exclusive |
| **OneTrust** | Commercial | Opaque — no rubric/model disclosed | Assessment/workflow-level, not published per-clause scoring | No | Unknown (marketed as "AI," specifics undisclosed) | Unknown | Marketed broadly, no public GDPR-specific technical detail |
| **TrustArc** | Commercial | Opaque; certification involves human assessors | Program/assessment-level | No | Primarily human-driven workflow tooling | Unknown | Marketed broadly ("Intelligent GDPR Assessments"), methodology undisclosed |
| **Osano** | Commercial | Opaque; some scan criteria disclosed at a high level (consent banners, dark patterns, GPC/IAB signals) | Site-level technical/consent-signal checks | No | Appears rule-based/scanning, not semantic NLP of clause text | No | Not GDPR-exclusive; consent-layer focus |
| **Termly / iubenda** | Commercial | Opaque; primarily template-matching for generation | Document-template level | No | Rule/template-based generation, not analysis of arbitrary existing text | No | Template libraries cover GDPR among other laws; not independent compliance judgment |
| **This project (proposed)** | Academic/applied | Designed to be published: retrieval trace + rubric-based judge | Article-level GDPR compliance scoring (target) | Open-source, fine-tuned open-weight judge (target) | LLM (fine-tuned), judge role | **Yes** — RAG over GDPR text | **Yes** — GDPR-exclusive by design |

*Note: several "Not confirmed" entries reflect genuine limits of what could be verified from publicly indexed abstracts and marketing pages in this review round, not a claim that no such disclosure exists anywhere (e.g., in a paper's appendix or artifact). These should be re-verified against full-text PDFs before being asserted as fact in the white paper.*

---

## 5. Gap analysis: what this project can credibly claim

Drawing on the academic and commercial landscape above, four gaps stand out as both genuine (evidenced by the sources reviewed) and realistically addressable by a RAG-over-GDPR-text-plus-fine-tuned-open-LLM-judge design:

**Gap 1 — No open, fine-tuned, GDPR-exclusive judge model exists.** Every LLM-based academic system found in the 2024–2026 window (PolicyGPT, the word-level transparency annotator, the USENIX Security '25 scale evaluation) relies on zero-/few-shot prompting of general-purpose, typically closed, commercial LLMs rather than a model fine-tuned specifically for GDPR compliance judgment. None ship an open-weight judge model. A fine-tuned open-source model — inspectable, locally deployable, rerunnable without API costs or vendor drift, and specifically tuned on GDPR-aligned judgments — is a concrete, currently-unclaimed position.

**Gap 2 — No dataset or system anchors output to explicit GDPR articles at fine granularity with public, reproducible methodology.** OPP-115 and APP-350 are law-agnostic by design (documented directly by the GDPR-mapping study in Section 3.1); PolicyIE aligns loosely with GDPR's four practice areas but not article-by-article, and is too small for fine-tuning; the closest recent academic work (2503.10727) targets only the transparency/disclosure requirements (Art. 13–14-style), not GDPR's fuller compliance surface (lawful basis under Art. 6, data subject rights under Art. 15–22, security under Art. 32, international transfers under Ch. V, etc.). A system whose RAG layer retrieves and cites the specific GDPR article(s) a clause is being judged against — and that publishes its retrieval + rubric methodology — occupies space no reviewed system fully covers.

**Gap 3 — The commercial market is opaque by default; nothing publishes a reproducible scoring methodology.** OneTrust, TrustArc, Osano, Termly, and iubenda were all found, in this review, to disclose marketing-level claims about AI or automated checking without publishing scoring rubrics, validation data, or accuracy benchmarks (Section 2). A project that publishes its RAG retrieval traces, its rubric, and its evaluation protocol against a stated GDPR annotation scheme is differentiable on **auditability and reproducibility** alone, independent of raw accuracy — this is a defensible, low-risk claim since it rests on documented absence of disclosure rather than a comparative performance claim.

**Gap 4 — No reviewed system combines RAG grounding in primary legal text with LLM-as-judge scoring for GDPR specifically.** RAG usage was confirmed only in the word-level transparency annotator (2503.10727), and there it retrieves within the requirement-annotation loop, not explicitly framed as "retrieval over GDPR statutory/recital text to ground a judge's citation of the legal basis for its verdict." Prior systems either classify practices (Polisis, PolicyGPT, PolicyIE) or judge against a fixed, hand-authored clause list distilled once from multiple laws (Xie et al.) rather than retrieving live from the regulation text per-judgment. Explicitly grounding each compliance verdict in retrieved GDPR article/recital passages — making the judge's citations inspectable and updatable if guidance changes — is a differentiator worth stating carefully rather than overclaiming, since the boundary with Xie et al.'s system in particular should be re-checked against that paper's full text before publication.

**Caveat for the white paper draft:** two of the works most likely to blunt these claims — the USENIX Security 2025 paper (artifact badge, possible partial code release; multi-law, GDPR included) and the word-level GDPR transparency paper (confirmed RAG usage) — were reviewed here at abstract/summary depth, not full-text depth. Before finalizing the differentiation section, both should be read in full (including their artifact appendices/supplementary code) to confirm neither already ships an open, fine-tuned GDPR judge model with article-level RAG grounding, which would narrow Gaps 1 and 4 respectively.

---

## Full source list

- [Polisis (USENIX Security 2018)](https://www.usenix.org/conference/usenixsecurity18/presentation/harkous)
- [Polisis (arXiv:1802.02561)](https://arxiv.org/abs/1802.02561)
- [PriBot / Polisis demo](https://pribot.org/)
- [Polisis unofficial reimplementation](https://github.com/quanmou/polisis)
- [PolicyLint (USENIX Security 2019)](https://www.usenix.org/conference/usenixsecurity19/presentation/andow)
- [PolicyLint full PDF](https://www.usenix.org/system/files/sec19-andow.pdf)
- [CLAUDETTE (Artificial Intelligence and Law, Springer)](https://link.springer.com/article/10.1007/s10506-019-09243-2)
- [CLAUDETTE meets GDPR (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3208596)
- [CLAUDETTE project site](http://claudette.eui.eu/)
- [PolicyGPT (arXiv:2309.10238)](https://arxiv.org/abs/2309.10238)
- [PolicyGPT (Hugging Face paper page)](https://huggingface.co/papers/2309.10238)
- [Word-level GDPR Transparency Annotation with LLMs (arXiv:2503.10727)](https://arxiv.org/abs/2503.10727)
- [Evaluating Privacy Policies under Modern Privacy Laws at Scale (USENIX Security 2025)](https://www.usenix.org/conference/usenixsecurity25/presentation/xie)
- [RegCheck: LLM-Driven GDPR Compliance Checking for Android Apps (FSE 2025)](https://jacquesklein2302.github.io/papers/2025-FSE-IVR-RegCheck.pdf)
- [LLM-assisted Extraction of Regulatory Requirements: GDPR case study](https://orbilu.uni.lu/bitstream/10993/65265/1/2025-RE-ACSBLSVS.pdf)
- [Large Language Models: a new approach for privacy policy analysis at scale (Computing, 2024)](https://link.springer.com/article/10.1007/s00607-024-01331-9)
- [OneTrust — Scale Privacy with AI](https://www.onetrust.com/scale-privacy-with-ai/)
- [OneTrust — Privacy Automation](https://www.onetrust.com/solutions/privacy-automation/)
- [TrustArc — Intelligent GDPR Assessments](https://www.prnewswire.com/news-releases/trustarc-launches-intelligent-gdpr-assessments-to-drive-compliance-automation-300612598.html)
- [TrustArc — GDPR Validation/Certification](https://trustarc.com/products/assurance-certifications/gdpr-validation/)
- [TrustArc — Assessment Manager](https://trustarc.com/products/privacy-data-governance/assessment-manager/)
- [Osano — Compliance Check](https://www.osano.com/features/compliance-check)
- [Osano — Compliance Check announcement](https://www.osano.com/updates/osano-compliance-check-automated-proof-website-privacy-compliance)
- [Termly](https://termly.io/)
- [Termly — laws covered](https://termly.io/resources/articles/which-laws-does-termly-cover/)
- [iubenda — GDPR compliance](https://www.iubenda.com/en/help/140769-gdpr-compliance-made-easy/)
- [iubenda — Privacy and Cookie Policy Generator](https://www.iubenda.com/en/privacy-and-cookie-policy-generator/)
- [OPP-115 corpus (Usable Privacy Policy Project)](https://usableprivacy.org/data)
- [Mapping the GDPR to a Privacy Policy Corpus Annotation Scheme](https://par.nsf.gov/servlets/purl/10257054)
- [MAPS / APP-350 (PoPETs 2019)](https://petsymposium.org/2019/files/papers/issue3/popets-2019-0037.pdf)
- [PrivaSeer (arXiv:2004.11131)](https://arxiv.org/abs/2004.11131)
- [PrivaSeer (ACL Anthology 2021)](https://aclanthology.org/2021.acl-long.532/)
- [PolicyIE (arXiv:2101.00123)](https://arxiv.org/abs/2101.00123)
- [PolicyIE GitHub repository](https://github.com/wasiahmad/PolicyIE)
