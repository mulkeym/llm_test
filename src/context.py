import random
from typing import List, Optional


class ContextGenerator:
    RUNBOOK_LINES = [
        "## Server Maintenance Procedure",
        "1. Verify backup completion status on the NAS dashboard.",
        "2. Check disk utilization across all partitions using df -h.",
        "3. Review failed login attempts in /var/log/auth.log for the past 24 hours.",
        "4. Rotate application logs older than 7 days using logrotate.",
        "5. Confirm NTP synchronization on all production servers.",
        "6. Verify SSL certificate expiration dates for public-facing services.",
        "7. Check RAID array status using mdadm --detail /dev/md0.",
        "8. Review cron job execution logs for any failures.",
        "9. Test failover for the primary database cluster.",
        "10. Update package lists and apply security patches.",
        "11. Verify firewall rules match the approved baseline.",
        "12. Check active connections on load balancers.",
        "13. Review memory utilization trends over the past week.",
        "14. Validate DNS resolution for all critical internal services.",
        "15. Confirm automated backups completed for all PostgreSQL databases.",
        "16. Check Docker container health status on all hosts.",
        "17. Review and rotate API keys older than 90 days.",
        "18. Verify monitoring alert thresholds are correctly configured.",
        "19. Test disaster recovery runbook with a tabletop exercise.",
        "20. Document any changes made during this maintenance window.",
        "### Pre-Maintenance Checklist",
        "- Notify stakeholders at least 24 hours in advance.",
        "- Create a snapshot of all VMs before patching.",
        "- Ensure rollback procedures are documented and tested.",
        "- Verify that the on-call engineer is available.",
        "### Post-Maintenance Validation",
        "- Run smoke tests against all critical endpoints.",
        "- Verify all services are reporting healthy in monitoring.",
        "- Check that no alerts were triggered during the window.",
        "- Update the maintenance log with actions taken.",
        "### Network Checks",
        "- Verify BGP peering status with upstream providers.",
        "- Check interface error counters on core switches.",
        "- Validate VLAN configuration on trunk ports.",
        "- Test connectivity between all data center zones.",
    ]

    LOG_LINES = [
        "Mar 15 08:23:41 web-prod-01 sshd[12345]: Accepted publickey for admin from 10.0.1.50",
        "Mar 15 08:23:42 web-prod-01 systemd[1]: Starting Apache HTTP Server...",
        "Mar 15 08:23:43 web-prod-01 apache2[12400]: AH00558: Could not reliably determine FQDN",
        "Mar 15 08:24:01 db-prod-01 postgresql[5432]: LOG: checkpoint starting: time",
        "Mar 15 08:24:02 db-prod-01 postgresql[5432]: LOG: checkpoint complete: wrote 156 buffers",
        "Mar 15 08:24:15 lb-prod-01 haproxy[8080]: 10.0.2.15:43210 [15/Mar/2026:08:24:15] frontend~ backend/web-prod-01 0/0/1/15/16 200 2456",
        "Mar 15 08:25:00 mon-prod-01 prometheus[9090]: level=info msg=\"Scrape completed\" target=web-prod-01",
        "Mar 15 08:25:01 web-prod-02 nginx[1234]: 10.0.3.20 - - [15/Mar/2026:08:25:01] \"GET /api/health HTTP/1.1\" 200 15",
        "Mar 15 08:25:30 auth-prod-01 sshd[14000]: Failed password for invalid user test from 203.0.113.50",
        "Mar 15 08:26:00 db-prod-02 mysqld[3306]: InnoDB: Buffer pool hit rate 998 / 1000",
        "Mar 15 08:26:15 cache-prod-01 redis[6379]: DB saved on disk",
        "Mar 15 08:26:30 queue-prod-01 rabbitmq[5672]: accepting AMQP connection <0.1234.0> (10.0.4.10:51234 -> 10.0.4.20:5672)",
        "Mar 15 08:27:00 web-prod-01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=198.51.100.25 DST=10.0.1.10 PROTO=TCP DPT=22",
        "Mar 15 08:27:15 dns-prod-01 named[53]: client @0x7f8b3c query: internal.example.com IN A +",
        "Mar 15 08:27:30 mail-prod-01 postfix/smtp[2525]: connect to smtp.example.com[93.184.216.34]:25: Connection timed out",
        "Mar 15 08:28:00 vpn-prod-01 openvpn[1194]: client-01/10.8.0.6 PUSH: Received control message",
        "Mar 15 08:28:15 backup-prod-01 rsync[9999]: sent 1,234,567 bytes received 456 bytes total size 1,234,567",
        "Mar 15 08:28:30 ci-prod-01 jenkins[8080]: Build #1234 completed: SUCCESS",
        "Mar 15 08:29:00 k8s-master-01 kube-apiserver[6443]: I0315 08:29:00 handler.go:153] GET /api/v1/nodes: (1.234ms) 200",
    ]

    CONFIG_LINES = [
        "server {",
        "    listen 80;",
        "    server_name internal.example.com;",
        "    location / {",
        "        proxy_pass http://backend:8080;",
        "        proxy_set_header Host $host;",
        "        proxy_set_header X-Real-IP $remote_addr;",
        "    }",
        "    location /health {",
        "        return 200 'OK';",
        "    }",
        "}",
        "# Upstream configuration",
        "upstream backend {",
        "    server 10.0.1.10:8080 weight=5;",
        "    server 10.0.1.11:8080 weight=3;",
        "    server 10.0.1.12:8080 backup;",
        "}",
        "# Rate limiting zone",
        "limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;",
        "# SSL configuration",
        "ssl_protocols TLSv1.2 TLSv1.3;",
        "ssl_ciphers HIGH:!aNULL:!MD5;",
        "ssl_prefer_server_ciphers on;",
        "# Logging",
        "access_log /var/log/nginx/access.log combined;",
        "error_log /var/log/nginx/error.log warn;",
        "# Firewall rules (iptables export)",
        "-A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT",
        "-A INPUT -p tcp --dport 80 -j ACCEPT",
        "-A INPUT -p tcp --dport 443 -j ACCEPT",
        "-A INPUT -p tcp --dport 5432 -s 10.0.1.0/24 -j ACCEPT",
        "-A INPUT -j DROP",
    ]

    EMAIL_LINES = [
        "From: admin@example.com",
        "To: ops-team@example.com",
        "Subject: Re: Production deployment planned for Friday",
        "",
        "Team,",
        "",
        "Quick update on the deployment plan:",
        "- We'll start the rolling restart at 02:00 UTC.",
        "- The canary instance will get traffic first for 15 minutes.",
        "- If error rates stay below 0.1%, we proceed to full rollout.",
        "- Rollback plan: revert to tag v2.3.1 and restart all pods.",
        "",
        "Let me know if anyone has concerns.",
        "",
        "---",
        "From: devops@example.com",
        "To: ops-team@example.com",
        "Subject: Re: Database migration status",
        "",
        "The migration script ran for 47 minutes. All 12 million rows migrated.",
        "Spot-checked 1000 records, all look correct.",
        "Index rebuild completed. Query performance is back to baseline.",
        "",
        "---",
        "From: security@example.com",
        "To: ops-team@example.com",
        "Subject: Vulnerability scan results - March",
        "",
        "Scan completed. 3 medium findings:",
        "1. OpenSSH 8.2 on jump-host-01 (upgrade to 9.x recommended)",
        "2. TLS 1.0 still enabled on legacy-app-01",
        "3. Default SNMP community string on switch-floor3",
        "",
        "No critical findings. Remediation tickets created.",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_filler(self, filler_type: str, target_tokens: int) -> str:
        lines_map = {
            "runbook": self.RUNBOOK_LINES,
            "log_dump": self.LOG_LINES,
            "config_dump": self.CONFIG_LINES,
            "email_thread": self.EMAIL_LINES,
        }
        source = lines_map.get(filler_type, self.RUNBOOK_LINES)
        target_chars = target_tokens * 4
        result_lines = []
        current_chars = 0
        while current_chars < target_chars:
            line = self.rng.choice(source)
            result_lines.append(line)
            current_chars += len(line) + 1
        return "\n".join(result_lines)

    def insert_needles(self, filler: str, needles: List[dict]) -> str:
        lines = filler.split("\n")
        sorted_needles = sorted(needles, key=lambda n: n["position"], reverse=True)
        for needle in sorted_needles:
            pos = int(needle["position"] * len(lines))
            pos = max(0, min(pos, len(lines)))
            lines.insert(pos, needle["text"])
        return "\n".join(lines)

    def build_context_document(
        self,
        filler_type: str,
        target_tokens: int,
        needles: List[dict],
    ) -> str:
        filler = self.generate_filler(filler_type, target_tokens)
        return self.insert_needles(filler, needles)
