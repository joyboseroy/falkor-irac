// ============================================================
// falkor-irac: FalkorDB Schema for Indian Legal Reasoning
// ============================================================
// Run this file once to set up indexes and constraints
// after connecting to your FalkorDB instance.
//
// Usage:
//   python graph_schema/setup_schema.py
// ============================================================


// ------------------------------------------------------------
// NODE INDEXES
// ------------------------------------------------------------

CREATE INDEX ON :Case(citation)
CREATE INDEX ON :Case(year)
CREATE INDEX ON :Case(court)
CREATE INDEX ON :Statute(name)
CREATE INDEX ON :Section(number)
CREATE INDEX ON :LegalIssue(text)
CREATE INDEX ON :ProceduralEvent(event_type)
CREATE INDEX ON :Judge(name)


// ------------------------------------------------------------
// SAMPLE SCHEMA DOCUMENTATION (Cypher comments)
// ------------------------------------------------------------

// Node: Case
// Properties:
//   citation      (string)  -- e.g. "(2012) 9 SCC 1"
//   name          (string)  -- e.g. "Sanjay Chandra v. CBI"
//   court         (string)  -- "Supreme Court" | "High Court" | "District Court"
//   year          (integer)
//   bench_size    (integer) -- number of judges
//   bench_type    (string)  -- "division" | "full" | "constitutional"
//   matter_type   (string)  -- "bail" | "service" | "constitutional" | "criminal" | "civil"
//   ildc_id       (string)  -- ILDC corpus identifier if available
//   summary       (string)  -- one-paragraph summary

// Node: Judge
// Properties:
//   name          (string)
//   court         (string)
//   tenure_start  (integer)
//   tenure_end    (integer)

// Node: Statute
// Properties:
//   name          (string)  -- e.g. "Code of Criminal Procedure, 1973"
//   short_name    (string)  -- e.g. "CrPC"
//   year          (integer)
//   repealed      (boolean)
//   repealed_year (integer)

// Node: Section
// Properties:
//   number        (string)  -- e.g. "437"
//   title         (string)  -- e.g. "When bail may be taken in case of non-bailable offence"
//   text          (string)  -- full statutory text
//   statute       (string)  -- parent statute name
//   repealed      (boolean)
//   repealed_by   (string)  -- citation of amending act if repealed

// Node: LegalIssue
// Properties:
//   text          (string)  -- the question before the court
//   issue_type    (string)  -- "constitutional" | "procedural" | "substantive" | "evidentiary"

// Node: Rule
// Properties:
//   text          (string)  -- the legal principle extracted
//   source        (string)  -- "precedent" | "statute" | "custom"

// Node: Argument
// Properties:
//   text          (string)
//   party         (string)  -- "petitioner" | "respondent"

// Node: ProceduralEvent
// Properties:
//   event_type    (string)  -- see EVENT_TYPES below
//   date          (string)  -- ISO date if available
//   court         (string)
//   outcome       (string)

// Procedural event types:
//   FIR_FILED | CHARGE_SHEET_SUBMITTED | BAIL_APPLICATION_FILED |
//   BAIL_GRANTED | BAIL_DENIED | BAIL_CANCELLED | APPEAL_FILED |
//   HEARING_HELD | HEARING_DELAYED | INTERIM_PROTECTION_GRANTED |
//   STAY_GRANTED | STAY_VACATED | JUDGMENT_RESERVED | JUDGMENT_DELIVERED |
//   SENTENCE_IMPOSED | ACQUITTAL | CONVICTION | RTI_FILED | RTI_RESPONSE

// Node: Outcome
// Properties:
//   text          (string)  -- the conclusion of the court
//   outcome_type  (string)  -- "allowed" | "dismissed" | "modified" | "remanded"

// Node: Jurisdiction
// Properties:
//   court         (string)
//   territory     (string)
//   level         (string)  -- "district" | "high_court" | "supreme_court"


// ------------------------------------------------------------
// RELATIONSHIP TYPES
// ------------------------------------------------------------

// (Case)-[:CITES]->(Case)
//   Properties: proposition (string) -- what the case is cited for

// (Case)-[:OVERRULES]->(Case)
//   Properties: year (integer)

// (Case)-[:DISTINGUISHES]->(Case)
//   Properties: basis (string) -- factual or legal basis for distinction

// (Case)-[:CONFLICTS_WITH]->(Case)
//   Properties:
//     conflict_type (string) -- "coordinate_bench" | "per_incuriam" | "distinguished"
//     unresolved    (boolean)

// (Case)-[:RESOLVED_BY]->(Case)
//   Properties:
//     resolution_type (string) -- "larger_bench" | "full_bench" | "constitutional_bench"

// (Case)-[:NARROWED_BY]->(Case)
//   Properties: basis (string)

// (Case)-[:AUTHORED_BY]->(Judge)

// (Case)-[:APPLIES_RULE]->(Rule)

// (Case)-[:DECIDES]->(LegalIssue)

// (Case)-[:SUPPORTS_ARGUMENT]->(Argument)

// (Case)-[:CITES_STATUTE]->(Section)
//   Properties: purpose (string) -- "relied_upon" | "distinguished" | "referred"

// (ProceduralEvent)-[:TRIGGERS]->(ProceduralEvent)
//   Properties: condition (string) -- legal condition that triggers the next event

// (ProceduralEvent)-[:PRECEDES]->(ProceduralEvent)
//   Properties: time_gap_days (integer)

// (Case)-[:INVOLVES_EVENT]->(ProceduralEvent)

// (Case)-[:RESULTS_IN]->(Outcome)

// (Section)-[:PART_OF]->(Statute)

// (Section)-[:AMENDED_BY]->(Section)
//   Properties: year (integer)
