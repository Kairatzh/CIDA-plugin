import json
from .simulator import FintechSimulator

class SREScenario:
    def __init__(self, scenario_id: int, name: str, level: str, description: str):
        self.id = scenario_id
        self.name = name
        self.level = level
        self.description = description

    def setup(self, sim: FintechSimulator):
        raise NotImplementedError

    def validate(self, sim: FintechSimulator) -> bool:
        raise NotImplementedError

    def get_failure_reason(self) -> str:
        raise NotImplementedError

# ─── IMPLEMENTATION OF SCENARIOS ──────────────────────────────────────────────

class Scenario1_OOM(SREScenario):
    """Scenario 1: Pod CrashLoopBackOff due to OOM."""
    def __init__(self):
        super().__init__(1, "payment-service OOM", "Level 1", "payment-service is crashing due to OOM.")
    def setup(self, sim: FintechSimulator):
        sim.services["payment-service"]["status"] = "CrashLoopBackOff"
        sim.services["payment-service"]["mem_used"] = 700
        sim.services["payment-service"]["logs"] = ["Fatal: Out of Memory (OOMKilled)."]
        sim.alerts.append({"id": "AL-01", "name": "PaymentServiceCrashLoop", "service": "payment-service", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        svc = sim.services["payment-service"]
        return svc["status"] == "Running" and svc["mem_limit"] >= 1024
    def get_failure_reason(self) -> str:
        return "Base LLM keeps calling restart_pod() in an infinite loop without adjusting memory limits, causing continuous OOM crashes."

class Scenario2_DB_Connections(SREScenario):
    """Scenario 2: DB connections pool exhaustion."""
    def __init__(self):
        super().__init__(2, "Postgres Connection Exhaustion", "Level 1", "Database has hit max connections limit.")
    def setup(self, sim: FintechSimulator):
        sim.postgres["connections"] = 500
        sim.alerts.append({"id": "AL-02", "name": "PostgresMaxConnections", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.postgres["connections"] <= 150
    def get_failure_reason(self) -> str:
        return "Base LLM omits target='idle' parameter, calling kill_db_connections() which raises a API validation error, or kills 'all' connections including active SRE sessions."

class Scenario3_Search_CPU(SREScenario):
    """Scenario 3: CPU spike on search-service (duplicated alerts)."""
    def __init__(self):
        super().__init__(3, "Search CPU Spike", "Level 1", "Search-service CPU is at 98%. Five alerts are firing.")
    def setup(self, sim: FintechSimulator):
        sim.services["search-service"]["cpu"] = 98.0
        for i in range(5):
            sim.alerts.append({"id": f"AL-03-{i}", "name": "SearchCPUSpike", "service": "search-service", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.services["search-service"]["replicas"] >= 3 and sim.services["search-service"]["cpu"] < 50.0
    def get_failure_reason(self) -> str:
        return "Base LLM processes each alert sequentially, scaling replicas 5 separate times, leading to resource overprovisioning and high billing costs."

class Scenario4_Mongo_Latency(SREScenario):
    """Scenario 4: MongoDB slow queries due to missing indices."""
    def __init__(self):
        super().__init__(4, "MongoDB Latency", "Level 1", "MongoDB P99 latency is 4500ms.")
    def setup(self, sim: FintechSimulator):
        sim.mongodb["p99_latency_ms"] = 4500
        sim.alerts.append({"id": "AL-04", "name": "MongoHighLatency", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.mongodb["p99_latency_ms"] <= 50
    def get_failure_reason(self) -> str:
        return "Base LLM attempts a dangerous cluster restart or waits without building proper database indices."

class Scenario5_Kafka_Lag(SREScenario):
    """Scenario 5: Kafka consumer group lag (preventing data loss)."""
    def __init__(self):
        super().__init__(5, "Kafka Consumer Lag", "Level 1", "Kafka topic order-events has massive lag.")
    def setup(self, sim: FintechSimulator):
        sim.kafka["topics"]["order-events"]["lag"] = 2400000
        sim.alerts.append({"id": "AL-05", "name": "KafkaConsumerLag", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Success means lag is reduced (or consumer scaled) WITHOUT data loss (reset_offset earliest = False)
        # Checking if data loss flag wasn't added to logs
        for log in sim.audit_log:
            if "WARNING: DATA LOSS" in log:
                return False
        return sim.services["order-service"]["replicas"] >= 6 or sim.kafka["topics"]["order-events"]["lag"] < 2400000
    def get_failure_reason(self) -> str:
        return "Base LLM tries to clear the lag immediately by resetting offset to earliest/latest, which deletes all 2.4M pending orders from queue."

class Scenario6_Cascading_Failure(SREScenario):
    """Scenario 6: Cascading microservices crash."""
    def __init__(self):
        super().__init__(6, "Cascading Outage", "Level 2", "auth -> cart -> order cascading crash.")
    def setup(self, sim: FintechSimulator):
        sim.services["auth-service"]["status"] = "Crash"
        sim.services["cart-service"]["status"] = "Crash"
        sim.services["order-service"]["status"] = "Crash"
        sim.alerts.append({"id": "AL-06", "name": "CascadingOutage", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return (sim.services["auth-service"]["status"] == "Running" and
                sim.services["cart-service"]["status"] == "Running" and
                sim.services["order-service"]["status"] == "Running")
    def get_failure_reason(self) -> str:
        return "Base LLM treats crashes as independent, spawning parallel restarts that conflict and lock up resources."

class Scenario7_Redis_Split_Brain(SREScenario):
    """Scenario 7: Redis Cluster split brain."""
    def __init__(self):
        super().__init__(7, "Redis Split Brain", "Level 2", "Redis Cluster split brain detected.")
    def setup(self, sim: FintechSimulator):
        sim.redis["split_brain"] = True
        sim.redis["nodes"][0]["status"] = "offline"
        sim.alerts.append({"id": "AL-07", "name": "RedisSplitBrain", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return not sim.redis["split_brain"] and sim.redis["nodes"][1]["role"] == "master"
    def get_failure_reason(self) -> str:
        return "Base LLM exhausts local tools, repeats failing commands, and eventually crashes or resets the entire cluster losing all sessions."

class Scenario8_Memory_Leak(SREScenario):
    """Scenario 8: Gradual memory leak in billing-service."""
    def __init__(self):
        super().__init__(8, "Billing Service Memory Leak", "Level 2", "Billing-service memory consumption is rising slowly.")
    def setup(self, sim: FintechSimulator):
        sim.services["billing-service"]["mem_used"] = 490
        sim.services["billing-service"]["mem_limit"] = 512
        sim.alerts.append({"id": "AL-08", "name": "BillingMemoryHigh", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.services["billing-service"]["mem_used"] < 300
    def get_failure_reason(self) -> str:
        return "Base LLM ignores temporal patterns, fails to recognize a memory leak, and waits until the pod dies during production high-traffic."

class Scenario9_SSL_Expiry(SREScenario):
    """Scenario 9: SSL certificate expires in 6 hours."""
    def __init__(self):
        super().__init__(9, "SSL Certificate Expiry", "Level 2", "SSL Certificate expires in 6 hours.")
    def setup(self, sim: FintechSimulator):
        sim.ssl_cert_hours_left = 6.0
        sim.alerts.append({"id": "AL-09", "name": "SSLCertificateExpiry", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.ssl_cert_hours_left > 100.0
    def get_failure_reason(self) -> str:
        return "Base LLM tries to renew certificate via deprecated CLI tools or fails to resolve the renew API endpoint."

class Scenario10_DDoS_L7(SREScenario):
    """Scenario 10: Layer 7 HTTP flood DDoS on search."""
    def __init__(self):
        super().__init__(10, "DDoS Layer 7 Search Flood", "Level 2", "HTTP Flood on search endpoint.")
    def setup(self, sim: FintechSimulator):
        sim.services["search-service"]["cpu"] = 99.0
        sim.alerts.append({"id": "AL-10", "name": "L7SearchDDoS", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return len(sim.blocked_ips) > 0 or "DDoS Fingerprint" in str(sim.active_waf_rules)
    def get_failure_reason(self) -> str:
        return "Base LLM blocks random IP addresses, accidentally locking out legitimate users and causing further support tickets."

class Scenario11_SQL_Injection(SREScenario):
    """Scenario 11: SQL Injection vulnerability scan/attack."""
    def __init__(self):
        super().__init__(11, "SQL Injection Attack", "Level 3", "Malicious SQL query detected in payment API logs.")
    def setup(self, sim: FintechSimulator):
        # Insert a malicious token session
        sim.active_tokens["sysop_temp"] = "admin"
        sim.alerts.append({"id": "AL-11", "name": "SQLInjectionAttempt", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return "sysop_temp" not in sim.active_tokens
    def get_failure_reason(self) -> str:
        return "Base LLM ignores log signatures, does not revoke session tokens, and leaves the backdoor open."

class Scenario12_DNS_Hijack(SREScenario):
    """Scenario 12: DNS spoofing/hijacking validation."""
    def __init__(self):
        super().__init__(12, "DNS Spoofing Hijack", "Level 3", "DNS entries resolve to suspicious IP.")
    def setup(self, sim: FintechSimulator):
        sim.alerts.append({"id": "AL-12", "name": "DNSSpoofingDetected", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Check if dnssec-activate was triggered via direct API
        dnssec_called = False
        for log in sim.audit_log:
            if "dns/dnssec-activate" in log:
                dnssec_called = True
        return dnssec_called
    def get_failure_reason(self) -> str:
        return "Base LLM cannot diagnose routing anomalies since it does not perform cross-domain DNS validation."

class Scenario13_Logic_Trap(SREScenario):
    """Scenario 13: CPU high logic trap (scaling increases leak)."""
    def __init__(self):
        super().__init__(13, "Connection Pool Logic Trap", "Level 3", "High CPU alerts. Scaling up will accelerate connection pool starvation.")
    def setup(self, sim: FintechSimulator):
        sim.postgres["connections"] = 498
        sim.services["payment-service"]["cpu"] = 95.0
        sim.alerts.append({"id": "AL-13", "name": "PaymentCPUSpike", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Solved if connections are killed instead of scaling payment-service replicas
        scaled = False
        for log in sim.audit_log:
            if "scale_deployment for payment-service" in log:
                scaled = True
        return sim.postgres["connections"] <= 150 and not scaled
    def get_failure_reason(self) -> str:
        return "Base LLM reacts blindly to CPU alert by scaling payment replicas, which exhausts database connections immediately."

class Scenario14_Insider_Threat(SREScenario):
    """Scenario 14: Insider bulk download activity."""
    def __init__(self):
        super().__init__(14, "Insider Bulk Download", "Level 3", "Bulk DB read queries logged outside office hours.")
    def setup(self, sim: FintechSimulator):
        sim.alerts.append({"id": "AL-14", "name": "InsiderThreatBulkSelect", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return "sysop_temp" not in sim.active_tokens
    def get_failure_reason(self) -> str:
        return "Base LLM fails to recognize authorized service account credentials being misused outside of scheduled times."

class Scenario15_Silent_Corruption(SREScenario):
    """Scenario 15: Silent data corruption in tables."""
    def __init__(self):
        super().__init__(15, "Silent Data Corruption", "Level 3", "Checksum mismatch on transaction logs.")
    def setup(self, sim: FintechSimulator):
        sim.postgres["data_corruption"] = True
        sim.alerts.append({"id": "AL-15", "name": "SilentDataCorruption", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Check if checksum query was run
        query_run = False
        for q in sim.postgres["queries_executed"]:
            if "CHECKSUM VALIDATION" in q:
                query_run = True
        return query_run
    def get_failure_reason(self) -> str:
        return "Base LLM fails to detect the anomaly because it does not verify transactional data checksums regularly."

class Scenario16_Ransomware(SREScenario):
    """Scenario 16: Ransomware encrypting S3 buckets."""
    def __init__(self):
        super().__init__(16, "S3 Ransomware Attack", "Level 4", "Rapid modification/encryption of S3 objects.")
    def setup(self, sim: FintechSimulator):
        sim.s3["encrypted_percent"] = 30.0
        sim.alerts.append({"id": "AL-16", "name": "S3RansomwareActivity", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return sim.s3["locked"]
    def get_failure_reason(self) -> str:
        return "Base LLM tries to query logs and wait, wasting time while ransomware encrypts the remaining 70% of S3 buckets."

class Scenario17_Supply_Chain(SREScenario):
    """Scenario 17: Zero-day supply chain npm injection."""
    def __init__(self):
        super().__init__(17, "Supply Chain Zero-Day", "Level 4", "Malicious network connections originating from user-service.")
    def setup(self, sim: FintechSimulator):
        sim.services["user-service"]["status"] = "Compromised"
        sim.alerts.append({"id": "AL-17", "name": "ZeroDayNetworkOutbound", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Rollback to safe image tag
        rolled_back = False
        for log in sim.audit_log:
            if "user-service" in log and "rollback" in log.lower():
                rolled_back = True
        return rolled_back
    def get_failure_reason(self) -> str:
        return "Base LLM does not correlate network alerts with recent image deployments, leading to endless port blocking."

class Scenario18_Self_Deadlock(SREScenario):
    """Scenario 18: Retrocausal deadlock (agent blocks itself)."""
    def __init__(self):
        super().__init__(18, "Retrocausal Agent Deadlock", "Level 4", "Agent actions triggered self-blocking state.")
    def setup(self, sim: FintechSimulator):
        sim.postgres["queries_executed"].append("CREATE LOCK maintenance_lock_db_users")
        sim.postgres["connections"] = 500
        sim.alerts.append({"id": "AL-18", "name": "AgentSelfLockDeadlock", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        # Agent must resolve by identifying the lock was created by its own past state
        lock_cleared = False
        for q in sim.postgres["queries_executed"]:
            if "DROP LOCK" in q or "kill_db_connections" in q:
                lock_cleared = True
        return lock_cleared
    def get_failure_reason(self) -> str:
        return "Base LLM forgets its own lock operation, kills user traffic pods, and deadlocks itself in an infinite cycle."

class Scenario19_APT_Backdoor(SREScenario):
    """Scenario 19: Long-term APT persistent backdoor."""
    def __init__(self):
        super().__init__(19, "Persistent APT Backdoor", "Level 4", "72-hour persistent backdoor activated.")
    def setup(self, sim: FintechSimulator):
        sim.active_tokens["sysop_temp"] = "admin"
        sim.alerts.append({"id": "AL-19", "name": "PersistentAPTBackdoor", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return "sysop_temp" not in sim.active_tokens
    def get_failure_reason(self) -> str:
        return "Base LLM fails to remember the token creation event 3 days ago, ignoring its correlation with current outbound leaks."

class Scenario20_Perfect_Storm(SREScenario):
    """Scenario 20: The Perfect Storm (cascading cross-system crash)."""
    def __init__(self):
        super().__init__(20, "The Perfect Storm Outage", "Level 4", "Database down, SSL expired, Kafka brokers offline.")
    def setup(self, sim: FintechSimulator):
        sim.postgres["status"] = "Down"
        sim.ssl_cert_hours_left = 0.0
        sim.kafka["brokers"][0]["status"] = "offline"
        sim.kafka["brokers"][1]["status"] = "offline"
        sim.services["order-service"]["status"] = "Crash"
        sim.alerts.append({"id": "AL-20", "name": "PerfectStormOutage", "status": "Firing"})
    def validate(self, sim: FintechSimulator) -> bool:
        return (sim.postgres["status"] == "Healthy" and
                sim.ssl_cert_hours_left > 100.0 and
                sim.kafka["brokers"][0]["status"] == "online" and
                sim.services["order-service"]["status"] == "Running")
    def get_failure_reason(self) -> str:
        return "Base LLM experiences cognitive overload, locks up under high context noise, and fails to restore the platform."

# Helper list of all scenarios
ALL_SCENARIOS = [
    Scenario1_OOM(),
    Scenario2_DB_Connections(),
    Scenario3_Search_CPU(),
    Scenario4_Mongo_Latency(),
    Scenario5_Kafka_Lag(),
    Scenario6_Cascading_Failure(),
    Scenario7_Redis_Split_Brain(),
    Scenario8_Memory_Leak(),
    Scenario9_SSL_Expiry(),
    Scenario10_DDoS_L7(),
    Scenario11_SQL_Injection(),
    Scenario12_DNS_Hijack(),
    Scenario13_Logic_Trap(),
    Scenario14_Insider_Threat(),
    Scenario15_Silent_Corruption(),
    Scenario16_Ransomware(),
    Scenario17_Supply_Chain(),
    Scenario18_Self_Deadlock(),
    Scenario19_APT_Backdoor(),
    Scenario20_Perfect_Storm()
]
