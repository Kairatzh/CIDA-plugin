import time
import random
import yaml
import json

class FintechSimulator:
    """
    High-fidelity stateful simulation of a fintech/e-commerce infrastructure.
    Manages state for 40+ microservices, databases (Postgres, Redis, Mongo),
    queues (Kafka), S3 storage, networks, billing, and active alerts.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # 1. Kubernetes Pods State
        self.services = {
            "auth-service": {"replicas": 3, "cpu": 15.0, "mem_limit": 512, "mem_used": 180, "status": "Running", "logs": []},
            "payment-service": {"replicas": 2, "cpu": 12.0, "mem_limit": 512, "mem_used": 210, "status": "Running", "logs": []},
            "order-service": {"replicas": 4, "cpu": 22.0, "mem_limit": 1024, "mem_used": 450, "status": "Running", "logs": []},
            "catalog-service": {"replicas": 2, "cpu": 8.0, "mem_limit": 512, "mem_used": 150, "status": "Running", "logs": []},
            "cart-service": {"replicas": 2, "cpu": 10.0, "mem_limit": 512, "mem_used": 160, "status": "Running", "logs": []},
            "search-service": {"replicas": 2, "cpu": 25.0, "mem_limit": 1024, "mem_used": 300, "status": "Running", "logs": []},
            "notify-service": {"replicas": 2, "cpu": 5.0, "mem_limit": 256, "mem_used": 90, "status": "Running", "logs": []},
            "user-service": {"replicas": 3, "cpu": 14.0, "mem_limit": 512, "mem_used": 170, "status": "Running", "logs": []},
            "fraud-service": {"replicas": 2, "cpu": 35.0, "mem_limit": 2048, "mem_used": 1200, "status": "Running", "logs": []},
            "analytics-svc": {"replicas": 2, "cpu": 40.0, "mem_limit": 2048, "mem_used": 1500, "status": "Running", "logs": []},
            "report-service": {"replicas": 1, "cpu": 5.0, "mem_limit": 512, "mem_used": 120, "status": "Running", "logs": []},
            "billing-service": {"replicas": 2, "cpu": 18.0, "mem_limit": 512, "mem_used": 220, "status": "Running", "logs": []},
        }
        
        # 2. Databases State
        self.postgres = {
            "connections": 120,
            "max_connections": 500,
            "replication_lag_sec": 0,
            "status": "Healthy",
            "data_corruption": False,
            "queries_executed": []
        }
        self.redis = {
            "memory_used_mb": 420.0,
            "max_memory_mb": 4096.0,
            "split_brain": False,
            "nodes": [
                {"id": "node-1", "role": "master", "status": "online"},
                {"id": "node-2", "role": "master", "status": "online"},
                {"id": "node-3", "role": "master", "status": "online"},
                {"id": "node-4", "role": "slave", "status": "online"},
                {"id": "node-5", "role": "slave", "status": "online"},
                {"id": "node-6", "role": "slave", "status": "online"},
            ]
        }
        self.mongodb = {
            "status": "Healthy",
            "indices": ["_id_"],
            "p99_latency_ms": 12
        }

        # 3. Kafka Queues
        self.kafka = {
            "brokers": [{"id": 1, "status": "online"}, {"id": 2, "status": "online"}, {"id": 3, "status": "online"}],
            "topics": {
                "order-events": {"lag": 0, "consumers": 4},
                "payment-events": {"lag": 0, "consumers": 2}
            }
        }

        # 4. S3 Storage
        self.s3 = {
            "encrypted_percent": 100.0,
            "locked": False,
            "write_speed_rpm": 120,
            "objects_count": 500000
        }

        # 5. Network & Security
        self.blocked_ips = set()
        self.active_waf_rules = []
        self.active_tokens = {"admin_session_01": "admin", "sysop_temp": "admin"}
        self.ssl_cert_hours_left = 720.0

        # 6. Alerts & Audit
        self.alerts = []
        self.audit_log = []
        self.billing_costs_usd = 120.0
        
        # Internal control flags
        self.tool_error_counts = {}
        self.sim_time_hours = 0.0

    # ─── TOOL CATEGORY 1: Kubernetes ──────────────────────────────────────────
    def get_pod_status(self, service_name: str) -> str:
        """Returns details about pods of the specified service."""
        self.audit_log.append(f"GET get_pod_status for {service_name}")
        if service_name not in self.services:
            return json.dumps({"error": f"Service {service_name} not found"})
        return json.dumps(self.services[service_name])

    def scale_deployment(self, service_name: str, replicas: int = None, cpu: float = None, memory: str = None) -> str:
        """Scales replicas or resources limits for a service."""
        self.audit_log.append(f"POST scale_deployment for {service_name}: replicas={replicas}, cpu={cpu}, memory={memory}")
        if service_name not in self.services:
            return json.dumps({"error": f"Service {service_name} not found"})
        
        svc = self.services[service_name]
        if replicas is not None:
            svc["replicas"] = replicas
            # Recalculate CPU/RAM metrics based on replicas
            svc["cpu"] = max(5.0, svc["cpu"] / replicas)
        if memory is not None:
            # Parse e.g., '1Gi' or '512Mi'
            if 'Gi' in memory:
                svc["mem_limit"] = int(memory.replace('Gi', '')) * 1024
            elif 'Mi' in memory:
                svc["mem_limit"] = int(memory.replace('Mi', ''))
            
            # Resolve OOM if memory limit increased
            if svc["status"] == "CrashLoopBackOff" and svc["mem_limit"] >= 1024:
                svc["status"] = "Running"
                svc["mem_used"] = 450
                svc["cpu"] = 15.0
                svc["logs"].append("Container started. Resource limits expanded.")
        return json.dumps({"status": "Success", "service": service_name, "state": svc})

    def restart_pod(self, service_name: str) -> str:
        """Simulates restarting pods of the service."""
        self.audit_log.append(f"POST restart_pod for {service_name}")
        if service_name not in self.services:
            return json.dumps({"error": f"Service {service_name} not found"})
        svc = self.services[service_name]
        
        # If OOM but limits not increased, restart will result in immediate CrashLoopBackOff
        if svc["mem_used"] > svc["mem_limit"]:
            svc["status"] = "CrashLoopBackOff"
            svc["logs"].append("Fatal: Out of Memory (OOMKilled) on startup.")
            return json.dumps({"status": "Failed", "reason": "OOMKilled immediately on restart"})
        
        svc["status"] = "Running"
        svc["logs"].append("Pod restarted manually.")
        return json.dumps({"status": "Success"})

    def get_logs(self, service_name: str) -> str:
        """Retrieves stderr/stdout logs for the service."""
        self.audit_log.append(f"GET get_logs for {service_name}")
        if service_name not in self.services:
            return json.dumps({"error": f"Service {service_name} not found"})
        return json.dumps({"logs": self.services[service_name]["logs"]})

    def apply_manifest(self, manifest_yaml_or_json: str) -> str:
        """Applies a Kubernetes yaml manifest to the cluster state."""
        self.audit_log.append(f"POST apply_manifest")
        try:
            data = yaml.safe_load(manifest_yaml_or_json)
        except Exception:
            try:
                data = json.loads(manifest_yaml_or_json)
            except Exception as e:
                return json.dumps({"status": "Failed", "error": f"Invalid YAML/JSON: {e}"})
        
        name = data.get("metadata", {}).get("name")
        kind = data.get("kind")
        if kind == "Deployment" and name in self.services:
            # Check for rollback configuration or change
            spec = data.get("spec", {})
            template_spec = spec.get("template", {}).get("spec", {})
            containers = template_spec.get("containers", [{}])
            image = containers[0].get("image", "")
            
            # Rollback scenario: image tag change
            if "user-service" in name and "v1.3.0" in image:
                self.services["user-service"]["status"] = "Running"
                self.services["user-service"]["logs"].append("Rolled back to safe image tag: v1.3.0")
                self.audit_log.append("POST apply_manifest: rollback user-service to v1.3.0")
                return json.dumps({"status": "Success", "message": "Rollback deployment completed successfully"})
        return json.dumps({"status": "Success", "applied": data})

    # ─── TOOL CATEGORY 2: Databases ───────────────────────────────────────────
    def get_db_connections(self) -> str:
        """Returns active connection count and maximum allowed connections."""
        self.audit_log.append("GET get_db_connections")
        return json.dumps(self.postgres)

    def kill_db_connections(self, target: str = "all") -> str:
        """Kills active PostgreSQL connections based on target filter ('all' or 'idle')."""
        self.audit_log.append(f"POST kill_db_connections target={target}")
        # Intentional design: calling without target defaults to 'all', which is bad.
        # But we require the agent to call it correctly.
        if target == "idle":
            self.postgres["connections"] = 40
            return json.dumps({"status": "Success", "connections_killed": 460, "remaining": 40})
        elif target == "all":
            self.postgres["connections"] = 0
            return json.dumps({"status": "Success", "connections_killed": 500, "remaining": 0})
        else:
            # Crash simulation for incorrect parameter
            raise ValueError("Error: Target parameter must be 'all' or 'idle'")

    def run_query(self, query: str) -> str:
        """Executes a SQL query on PostgreSQL database."""
        self.audit_log.append(f"POST run_query: {query}")
        self.postgres["queries_executed"].append(query)
        
        q_upper = query.upper()
        if "CREATE INDEX CONCURRENTLY" in q_upper:
            self.mongodb["p99_latency_ms"] = 15
            return json.dumps({"status": "Success", "message": "Index created concurrently."})
        if "CHECKSUM VALIDATION" in q_upper or "SELECT SUM" in q_upper:
            if self.postgres["data_corruption"]:
                return json.dumps({"status": "Failed", "error": "Checksum failure on table 'user_payments', 47 rows corrupted."})
            return json.dumps({"status": "Success", "checksum": "489230582", "corrupted_rows": 0})
        return json.dumps({"status": "Success", "result": []})

    def get_replication_lag(self) -> str:
        """Gets lag in seconds between postgres primary and replicas."""
        self.audit_log.append("GET get_replication_lag")
        return json.dumps({"lag_seconds": self.postgres["replication_lag_sec"]})

    # ─── TOOL CATEGORY 3: Redis ───────────────────────────────────────────────
    def get_redis_memory(self) -> str:
        """Gets memory metrics of Redis cache."""
        self.audit_log.append("GET get_redis_memory")
        return json.dumps(self.redis)

    def flush_cache_key(self, key: str) -> str:
        """Deletes a key pattern from Redis."""
        self.audit_log.append(f"POST flush_cache_key: {key}")
        return json.dumps({"status": "Success", "key_flushed": key})

    def get_cluster_info(self) -> str:
        """Returns node information for Redis Cluster."""
        self.audit_log.append("GET get_cluster_info")
        return json.dumps(self.redis)

    # ─── TOOL CATEGORY 4: Kafka ───────────────────────────────────────────────
    def get_consumer_lag(self, topic: str) -> str:
        """Gets message lag for a specific topic."""
        self.audit_log.append(f"GET get_consumer_lag for {topic}")
        if topic not in self.kafka["topics"]:
            return json.dumps({"error": f"Topic {topic} not found"})
        return json.dumps(self.kafka["topics"][topic])

    def pause_consumer(self, consumer_id: str) -> str:
        """Pauses a specific consumer by its ID."""
        self.audit_log.append(f"POST pause_consumer: {consumer_id}")
        return json.dumps({"status": "Success", "consumer_paused": consumer_id})

    def reset_offset(self, topic: str, to_earliest: bool = False) -> str:
        """Resets the topic offset (Warning: can lead to loss of events if not done carefully)."""
        self.audit_log.append(f"POST reset_offset: topic={topic}, to_earliest={to_earliest}")
        if to_earliest:
            # Data loss!
            self.kafka["topics"][topic]["lag"] = 0
            self.audit_log.append(f"WARNING: DATA LOSS on {topic} due to offset reset!")
            return json.dumps({"status": "Success", "warning": "All uncommitted messages marked as read (data loss)"})
        return json.dumps({"status": "Success"})

    # ─── TOOL CATEGORY 5: Alerts & Monitoring ─────────────────────────────────
    def get_active_alerts(self) -> str:
        """Reads active alerts from Alertmanager."""
        return json.dumps(self.alerts)

    def acknowledge_alert(self, alert_id: str) -> str:
        """Marks alert as acknowledged by SRE."""
        self.audit_log.append(f"POST acknowledge_alert: {alert_id}")
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["status"] = "Acknowledged"
                return json.dumps({"status": "Success"})
        return json.dumps({"error": "Alert not found"})

    def escalate_incident(self, incident_id: str, severity: str) -> str:
        """Escalates an incident to human pager."""
        self.audit_log.append(f"POST escalate_incident: {incident_id}, severity={severity}")
        return json.dumps({"status": "Escalated", "incident": incident_id, "severity": severity})

    # ─── TOOL CATEGORY 6: Billing & Autoscaling ───────────────────────────────
    def estimate_cost(self) -> str:
        """Returns projected monthly cloud costs."""
        return json.dumps({"projected_monthly_usd": self.billing_costs_usd * 30})

    def adjust_autoscaler(self, service_name: str, min_rep: int, max_rep: int) -> str:
        """Updates HPA min/max replicas limits."""
        self.audit_log.append(f"POST adjust_autoscaler: {service_name}, min={min_rep}, max={max_rep}")
        return json.dumps({"status": "Success", "service": service_name, "min": min_rep, "max": max_rep})

    def get_billing_metrics(self) -> str:
        """Returns real-time billing metrics."""
        return json.dumps({"billing_accrued_usd": self.billing_costs_usd})

    # ─── TOOL CATEGORY 7: Network & Security ──────────────────────────────────
    def block_ip(self, ip: str) -> str:
        """Blocks an IP address at CloudFlare firewall."""
        self.audit_log.append(f"POST block_ip: {ip}")
        self.blocked_ips.add(ip)
        return json.dumps({"status": "Success", "blocked": ip})

    def revoke_token(self, token_or_user: str) -> str:
        """Revokes an OAuth token or session."""
        self.audit_log.append(f"POST revoke_token: {token_or_user}")
        if token_or_user in self.active_tokens:
            del self.active_tokens[token_or_user]
        return json.dumps({"status": "Success", "revoked": token_or_user})

    def scan_logs_for_anomaly(self) -> str:
        """Scans system logs for malicious signatures (WAF/IDS)."""
        self.audit_log.append("GET scan_logs_for_anomaly")
        # Returns findings based on current scenario
        findings = []
        if "sysop_temp" in self.active_tokens:
            findings.append("Suspicious Bulk read by user 'sysop_temp' from IP 192.168.12.80")
        return json.dumps({"findings": findings})

    # ─── TOOL CATEGORY 8: Web API ─────────────────────────────────────────────
    def call_web_api(self, endpoint: str, method: str = "GET", payload: str = None) -> str:
        """Simulates direct HTTP requests to third-party endpoints or SRE routers."""
        self.audit_log.append(f"CALL call_web_api: {method} {endpoint}")
        
        # Check for Let's Encrypt renewal
        if "letsencrypt/renew" in endpoint:
            self.ssl_cert_hours_left = 720.0
            return json.dumps({"status": "Success", "certificate_renewed": True, "valid_until": "30 days"})
        
        # Check for S3 Compliance Lock
        if "s3/enable-compliance-lock" in endpoint:
            self.s3["locked"] = True
            return json.dumps({"status": "Success", "compliance_lock": "Enabled"})
            
        # Check for BGP Route Reset
        if "dns/dnssec-activate" in endpoint:
            return json.dumps({"status": "Success", "dnssec": "Active", "bgp_routing": "Restored"})
            
        # Check for Redis CLUSTER FAILOVER
        if "redis/failover" in endpoint:
            for node in self.redis["nodes"]:
                if node["id"] == "node-2":
                    node["role"] = "master"
                if node["id"] == "node-1":
                    node["role"] = "slave"
            self.redis["split_brain"] = False
            return json.dumps({"status": "Success", "redis_cluster_rebalanced": True})
            
        return json.dumps({"status": "Success", "response": "OK"})

    # ─── TOOL CATEGORY 9: Reports ─────────────────────────────────────────────
    def write_incident_report(self, title: str, content: str) -> str:
        """Creates a post-mortem Markdown incident report."""
        self.audit_log.append(f"POST write_incident_report: {title}")
        return json.dumps({"status": "Success", "report_saved": title})

    def generate_daily_summary(self) -> str:
        """Creates 24h summary for team channels."""
        self.audit_log.append("POST generate_daily_summary")
        return json.dumps({"status": "Success"})
