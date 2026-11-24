# Copyright (c) 2026 Alfan Presekal (apresekal)
# SPDX-License-Identifier: Apache-2.0
"""Packet schema and parsing helpers used by the TCV 5G-NIDD experiments.

Only the required helpers are retained from the original preparation module.
Use T2_0b_relabel_5gnidd.py followed by T2_0c_fix_labels.py to prepare data.
"""

import numpy as np


def _import_scapy():
    global PcapReader, Ether, IP, IPv6, TCP, UDP, ICMP, SCTP, Raw
    from scapy.all import PcapReader, Ether, IP, IPv6, TCP, UDP, ICMP, SCTP, Raw


def build_packet_dtype(N: int) -> np.dtype:
    return np.dtype(
        [
            ("dataset", "S16"),
            ("pcap_id", "S96"),
            ("packet_idx", "<i8"),
            ("timestamp", "<f8"),
            ("beta", "<f8"),
            ("src_mac", "S18"),
            ("dst_mac", "S18"),
            ("src_ip", "S40"),
            ("dst_ip", "S40"),
            ("ip_version", "<i1"),
            ("ip_ttl", "<i2"),
            ("ip_id", "<i4"),
            ("frame_len", "<i4"),
            ("l4_proto", "<i2"),
            ("src_port", "<i4"),
            ("dst_port", "<i4"),
            ("tcp_flags", "<i2"),
            ("has_gtp", "?"),
            ("gtp_teid", "<i8"),
            ("has_sctp", "?"),
            ("l7_label", "S16"),
            ("binary_label", "S8"),
            ("attack_type", "S48"),
            ("attack_category", "S16"),
            ("normal_type", "S24"),
            ("payload_len_raw", "<i4"),
            ("payload", f"({N},)u1"),
        ]
    )


def normal_type_of(l4_proto, dst_port, has_gtp):
    if has_gtp and dst_port == 2152:
        return "gtp_user"
    if dst_port == 2123:
        return "gtp_control"
    if dst_port in (80, 8080, 7777, 3000):
        return "web"
    if dst_port in (443, 8443):
        return "tls"
    if dst_port == 53:
        return "dns"
    if dst_port == 8805:
        return "signaling"
    if dst_port == 830:
        return "oam"
    if l4_proto == 1:
        return "ping"
    if dst_port in (5060, 5061):
        return "voice"
    if dst_port in (1883, 8883):
        return "iot_telemetry"
    return "other"


def l7_label_of(l4_proto, dst_port, has_gtp):
    if has_gtp:
        return "gtp"
    if l4_proto == 1:
        return "icmp"
    if l4_proto == 17 and dst_port == 53:
        return "dns"
    if l4_proto == 17 and dst_port == 8805:
        return "pfcp"
    if l4_proto == 17 and dst_port in (2152, 2123):
        return "gtp"
    if l4_proto == 6 and dst_port in (80, 8080, 7777, 3000):
        return "http"
    if l4_proto == 6 and dst_port in (443, 8443):
        return "https"
    if l4_proto == 6 and dst_port == 830:
        return "netconf"
    if l4_proto == 132:
        return "sctp"
    if l4_proto == 6:
        return f"tcp_{dst_port}"
    if l4_proto == 17:
        return f"udp_{dst_port}"
    return "other"


def packet_to_row(
    pkt,
    dataset,
    pcap_id,
    packet_idx,
    beta,
    N,
    attack_type,
    attack_category,
    binary_label,
):
    src_mac = pkt[Ether].src if pkt.haslayer(Ether) else ""
    dst_mac = pkt[Ether].dst if pkt.haslayer(Ether) else ""
    src_ip = dst_ip = ""
    ip_version, ip_ttl, ip_id = -1, -1, -1
    l4_proto = 0
    if pkt.haslayer(IP):
        ip = pkt[IP]
        src_ip, dst_ip = ip.src, ip.dst
        ip_version, ip_ttl, ip_id = 4, int(ip.ttl), int(ip.id)
        l4_proto = int(ip.proto)
    elif pkt.haslayer(IPv6):
        ip6 = pkt[IPv6]
        src_ip, dst_ip = ip6.src, ip6.dst
        ip_version, ip_ttl = 6, int(ip6.hlim)
        l4_proto = int(ip6.nh)
    src_port = dst_port = -1
    tcp_flags = 0
    payload_bytes = b""
    if pkt.haslayer(TCP):
        t = pkt[TCP]
        src_port, dst_port = int(t.sport), int(t.dport)
        tcp_flags = int(t.flags)
        payload_bytes = bytes(t.payload)
        l4_proto = 6
    elif pkt.haslayer(UDP):
        u = pkt[UDP]
        src_port, dst_port = int(u.sport), int(u.dport)
        payload_bytes = bytes(u.payload)
        l4_proto = 17
    elif pkt.haslayer(ICMP):
        payload_bytes = bytes(pkt[ICMP].payload)
        l4_proto = 1
    elif pkt.haslayer(SCTP):
        s = pkt[SCTP]
        src_port, dst_port = int(s.sport), int(s.dport)
        payload_bytes = bytes(s.payload)
        l4_proto = 132
    has_gtp = l4_proto == 17 and dst_port in (2152, 2123)
    has_sctp = pkt.haslayer(SCTP)
    l7 = l7_label_of(l4_proto, dst_port, has_gtp)
    normal_t = ""
    if binary_label == "normal":
        normal_t = normal_type_of(l4_proto, dst_port, has_gtp)
    payload_len_raw = len(payload_bytes)
    payload_arr = np.zeros(N, dtype=np.uint8)
    if payload_len_raw > 0:
        b = payload_bytes[:N]
        payload_arr[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    return (
        dataset.encode()[:16],
        pcap_id.encode()[:96],
        packet_idx,
        float(pkt.time),
        beta,
        src_mac.encode()[:18],
        dst_mac.encode()[:18],
        src_ip.encode()[:40],
        dst_ip.encode()[:40],
        ip_version,
        ip_ttl,
        ip_id,
        int(len(pkt)),
        l4_proto,
        src_port,
        dst_port,
        tcp_flags,
        has_gtp,
        -1,
        has_sctp,
        l7.encode()[:16],
        binary_label.encode()[:8],
        attack_type.encode()[:48],
        attack_category.encode()[:16],
        normal_t.encode()[:24],
        payload_len_raw,
        payload_arr,
    )
