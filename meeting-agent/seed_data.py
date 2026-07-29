from ingest import ingest_transcript

t1 = """
Attendees: Sai, Priya, Rahul
Date: 2025-05-01

We decided to migrate from REST to gRPC for internal services by Q3.
Sai will benchmark gRPC vs REST latency by next Friday.
Priya will update the API design doc by end of month.
We will NOT use GraphQL — too much overhead for our use case.
"""

t2 = """
Attendees: Sai, Rahul, Kiran
Date: 2025-05-10

Revisited the API decision. Rahul suggested GraphQL for the mobile team.
Decided to keep gRPC for internal, evaluate GraphQL only for mobile BFF.
Kiran to prototype GraphQL BFF by next Wednesday.
Sai's gRPC benchmark is overdue — needs to be done this week.
"""

t3 = """
Attendees: Sai, Priya, Ananya
Date: 2025-05-20

Decided to delay Q3 gRPC migration to Q4 due to resource constraints.
Ananya joins as tech lead for the migration.
Priya's API design doc approved — no changes needed.
New decision: all new services will use gRPC from today itself.
"""

if __name__ == "__main__":
    ingest_transcript(t1, "2025-05-01", "Architecture Planning")
    ingest_transcript(t2, "2025-05-10", "API Strategy Review")
    ingest_transcript(t3, "2025-05-20", "Q3 Planning Sync")
    print("\nAll transcripts indexed.")