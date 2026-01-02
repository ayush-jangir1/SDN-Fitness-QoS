#!/usr/bin/env python3
"""
SDN Fitness Center Controller - Ryu Application
================================================
This Ryu controller application manages traffic flows in the fitness center
network, implementing QoS-based prioritization and health alert handling.

Key Features:
1. Dynamic flow management with OpenFlow 1.3
2. QoS-based traffic prioritization for injury recovery clients
3. Emergency traffic handling for heart rate spike alerts
4. Real-time flow statistics monitoring
5. REST API for external integration

"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
from ryu.lib import hub
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json
import time
import logging
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FitnessCenterController')


# REST API instance name
fitness_center_instance_name = 'fitness_center_api'


class FitnessCenterController(app_manager.RyuApp):
    """
    Main Ryu Controller Application for SDN-based Fitness Center.
    
    This controller implements:
    - Learning switch functionality
    - QoS-based flow prioritization
    - Health emergency detection and response
    - Traffic statistics collection
    - REST API for management
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    # Traffic priority levels (Queue IDs)
    PRIORITY_BEST_EFFORT = 0
    PRIORITY_INJURY_RECOVERY = 1
    PRIORITY_EMERGENCY = 2
    
    # Port definitions (UDP ports for fitness data)
    PORT_HEARTRATE = 5001
    PORT_STEPS = 5002
    PORT_WORKOUT = 5003
    PORT_EMERGENCY = 5004
    
    # Client configuration
    CLIENT_CONFIG = {
        '10.0.1.1': {'name': 'Client1', 'priority': True, 'condition': 'injury_recovery'},
        '10.0.1.2': {'name': 'Client2', 'priority': False, 'condition': 'normal'},
        '10.0.1.3': {'name': 'Client3', 'priority': False, 'condition': 'normal'},
        '10.0.1.4': {'name': 'Client4', 'priority': False, 'condition': 'normal'},
    }
    
    DASHBOARD_IP = '10.0.2.1'
    
    def __init__(self, *args, **kwargs):
        super(FitnessCenterController, self).__init__(*args, **kwargs)
        
        # MAC address table: {dpid: {mac: port}}
        self.mac_to_port = {}
        
        # Flow statistics
        self.flow_stats = defaultdict(lambda: defaultdict(dict))
        
        # Traffic metrics for analysis
        self.traffic_metrics = {
            'total_packets': 0,
            'priority_packets': 0,
            'emergency_packets': 0,
            'normal_packets': 0,
            'bytes_transmitted': 0,
            'latency_samples': [],
            'start_time': time.time()
        }
        
        # Emergency alerts log
        self.emergency_alerts = []
        
        # Datapath references
        self.datapaths = {}
        
        # Start statistics monitoring thread
        self.monitor_thread = hub.spawn(self._monitor_stats)
        
        # REST API setup
        wsgi = kwargs['wsgi']
        wsgi.register(FitnessCenterRestController, {fitness_center_instance_name: self})
        
        logger.info("Fitness Center Controller initialized")
        logger.info(f"Priority client (injury recovery): {self._get_priority_client()}")
    
    def _get_priority_client(self):
        """Get the IP of the priority (injury recovery) client."""
        for ip, config in self.CLIENT_CONFIG.items():
            if config['priority']:
                return ip
        return None
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handle switch connection and install initial flows.
        
        This method is called when a switch connects to the controller.
        It installs the table-miss flow entry that sends unmatched packets
        to the controller.
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        dpid = datapath.id
        self.datapaths[dpid] = datapath
        
        logger.info(f"Switch connected: dpid={dpid:016x}")
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions, table_id=0)
        
        # Install QoS-aware flows for known clients
        self._install_qos_flows(datapath)
        
        logger.info(f"Initial flows installed on switch dpid={dpid:016x}")
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """Handle switch state changes (connect/disconnect)."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                logger.info(f"Switch registered: dpid={datapath.id:016x}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                logger.info(f"Switch disconnected: dpid={datapath.id:016x}")
                del self.datapaths[datapath.id]
    
    def _install_qos_flows(self, datapath):
        """
        Install QoS-aware flows for priority traffic handling.
        
        Flow Rules:
        1. Emergency traffic (port 5004) -> Queue 2 (highest priority)
        2. Priority client traffic -> Queue 1 (high priority)
        3. Normal traffic -> Queue 0 (best effort)
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        priority_client_ip = self._get_priority_client()
        
        # Rule 1: Emergency traffic gets highest priority
        # Match UDP traffic on emergency port from any client
        for client_ip in self.CLIENT_CONFIG.keys():
            match = parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ip_proto=17,  # UDP
                ipv4_src=client_ip,
                udp_dst=self.PORT_EMERGENCY
            )
            # Action: Set queue 2 (emergency) and forward
            actions = [
                parser.OFPActionSetQueue(self.PRIORITY_EMERGENCY),
                parser.OFPActionOutput(ofproto.OFPP_NORMAL)
            ]
            self._add_flow(datapath, 100, match, actions)
            logger.info(f"Installed emergency flow for {client_ip}")
        
        # Rule 2: Priority client (injury recovery) gets high priority
        if priority_client_ip:
            for port in [self.PORT_HEARTRATE, self.PORT_STEPS, self.PORT_WORKOUT]:
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ip_proto=17,  # UDP
                    ipv4_src=priority_client_ip,
                    udp_dst=port
                )
                actions = [
                    parser.OFPActionSetQueue(self.PRIORITY_INJURY_RECOVERY),
                    parser.OFPActionOutput(ofproto.OFPP_NORMAL)
                ]
                self._add_flow(datapath, 50, match, actions)
            logger.info(f"Installed priority flows for injury recovery client: {priority_client_ip}")
        
        # Rule 3: Normal clients get best effort
        for client_ip, config in self.CLIENT_CONFIG.items():
            if not config['priority']:
                for port in [self.PORT_HEARTRATE, self.PORT_STEPS, self.PORT_WORKOUT]:
                    match = parser.OFPMatch(
                        eth_type=ether_types.ETH_TYPE_IP,
                        ip_proto=17,  # UDP
                        ipv4_src=client_ip,
                        udp_dst=port
                    )
                    actions = [
                        parser.OFPActionSetQueue(self.PRIORITY_BEST_EFFORT),
                        parser.OFPActionOutput(ofproto.OFPP_NORMAL)
                    ]
                    self._add_flow(datapath, 10, match, actions)
                logger.info(f"Installed best-effort flows for normal client: {client_ip}")
    
    def _add_flow(self, datapath, priority, match, actions, table_id=0, 
                  idle_timeout=0, hard_timeout=0, buffer_id=None):
        """
        Add a flow entry to the switch.
        
        Args:
            datapath: Switch datapath
            priority: Flow priority (higher = more important)
            match: Match criteria
            actions: Actions to perform
            table_id: Flow table ID
            idle_timeout: Idle timeout in seconds
            hard_timeout: Hard timeout in seconds
            buffer_id: Buffer ID (if packet is buffered)
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        instructions = [
            parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
        ]
        
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=instructions,
                table_id=table_id,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=instructions,
                table_id=table_id,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout
            )
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle packets sent to controller (table-miss).
        
        This implements learning switch functionality and collects
        traffic metrics for analysis.
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Update MAC table
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        # Check for IP packets to apply QoS
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        queue_id = self.PRIORITY_BEST_EFFORT
        
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            
            # Update traffic metrics
            self.traffic_metrics['total_packets'] += 1
            self.traffic_metrics['bytes_transmitted'] += len(msg.data)
            
            # Check for UDP packets (fitness data)
            udp_pkt = pkt.get_protocol(udp.udp)
            if udp_pkt:
                dst_port = udp_pkt.dst_port
                
                # Determine queue based on traffic type
                if dst_port == self.PORT_EMERGENCY:
                    queue_id = self.PRIORITY_EMERGENCY
                    self.traffic_metrics['emergency_packets'] += 1
                    self._log_emergency_alert(src_ip, msg.data)
                elif src_ip in self.CLIENT_CONFIG and self.CLIENT_CONFIG[src_ip]['priority']:
                    queue_id = self.PRIORITY_INJURY_RECOVERY
                    self.traffic_metrics['priority_packets'] += 1
                else:
                    queue_id = self.PRIORITY_BEST_EFFORT
                    self.traffic_metrics['normal_packets'] += 1
        
        # Build actions with queue assignment
        actions = [
            parser.OFPActionSetQueue(queue_id),
            parser.OFPActionOutput(out_port)
        ]
        
        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, 1, match, actions, 
                              idle_timeout=60, buffer_id=msg.buffer_id)
                return
            else:
                self._add_flow(datapath, 1, match, actions, idle_timeout=60)
        
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)
    
    def _log_emergency_alert(self, src_ip, data):
        """Log emergency alert for analysis."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'client_ip': src_ip,
            'client_name': self.CLIENT_CONFIG.get(src_ip, {}).get('name', 'Unknown'),
            'type': 'HEART_RATE_SPIKE'
        }
        self.emergency_alerts.append(alert)
        logger.warning(f"EMERGENCY ALERT: {alert}")
    
    def _monitor_stats(self):
        """Periodically request flow statistics from switches."""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)  # Request stats every 5 seconds
    
    def _request_stats(self, datapath):
        """Request flow and port statistics from a switch."""
        parser = datapath.ofproto_parser
        
        # Request flow stats
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        
        # Request port stats
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        self.flow_stats[dpid] = {}
        
        for stat in body:
            flow_info = {
                'priority': stat.priority,
                'match': str(stat.match),
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec
            }
            flow_key = f"flow_{stat.priority}_{hash(str(stat.match)) % 10000}"
            self.flow_stats[dpid][flow_key] = flow_info
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        for stat in body:
            port_info = {
                'port_no': stat.port_no,
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped': stat.tx_dropped,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors
            }
            self.flow_stats[dpid][f"port_{stat.port_no}"] = port_info
    
    def get_metrics(self):
        """Get current traffic metrics."""
        runtime = time.time() - self.traffic_metrics['start_time']
        return {
            'runtime_seconds': runtime,
            'total_packets': self.traffic_metrics['total_packets'],
            'priority_packets': self.traffic_metrics['priority_packets'],
            'emergency_packets': self.traffic_metrics['emergency_packets'],
            'normal_packets': self.traffic_metrics['normal_packets'],
            'bytes_transmitted': self.traffic_metrics['bytes_transmitted'],
            'packets_per_second': self.traffic_metrics['total_packets'] / max(1, runtime),
            'emergency_alerts': len(self.emergency_alerts)
        }
    
    def get_flow_stats(self):
        """Get current flow statistics."""
        return dict(self.flow_stats)
    
    def get_emergency_alerts(self):
        """Get list of emergency alerts."""
        return self.emergency_alerts
    
    def update_client_priority(self, client_ip, is_priority):
        """
        Dynamically update client priority status.
        
        This demonstrates SDN's ability to dynamically modify network behavior.
        """
        if client_ip in self.CLIENT_CONFIG:
            old_priority = self.CLIENT_CONFIG[client_ip]['priority']
            self.CLIENT_CONFIG[client_ip]['priority'] = is_priority
            
            # Reinstall flows on all switches
            for dp in self.datapaths.values():
                self._install_qos_flows(dp)
            
            logger.info(f"Updated priority for {client_ip}: {old_priority} -> {is_priority}")
            return True
        return False


class FitnessCenterRestController(ControllerBase):
    """
    REST API Controller for Fitness Center SDN Application.
    
    Endpoints:
    - GET /fitness/stats: Get traffic metrics
    - GET /fitness/flows: Get flow statistics
    - GET /fitness/alerts: Get emergency alerts
    - POST /fitness/priority: Update client priority
    """
    
    def __init__(self, req, link, data, **config):
        super(FitnessCenterRestController, self).__init__(req, link, data, **config)
        self.fitness_center_app = data[fitness_center_instance_name]
    
    @route('fitness', '/fitness/stats', methods=['GET'])
    def get_stats(self, req, **kwargs):
        """Get current traffic metrics."""
        metrics = self.fitness_center_app.get_metrics()
        body = json.dumps(metrics, indent=2)
        return Response(content_type='application/json', body=body)
    
    @route('fitness', '/fitness/flows', methods=['GET'])
    def get_flows(self, req, **kwargs):
        """Get current flow statistics."""
        flows = self.fitness_center_app.get_flow_stats()
        body = json.dumps(flows, indent=2, default=str)
        return Response(content_type='application/json', body=body)
    
    @route('fitness', '/fitness/alerts', methods=['GET'])
    def get_alerts(self, req, **kwargs):
        """Get emergency alerts."""
        alerts = self.fitness_center_app.get_emergency_alerts()
        body = json.dumps(alerts, indent=2)
        return Response(content_type='application/json', body=body)
    
    @route('fitness', '/fitness/priority', methods=['POST'])
    def update_priority(self, req, **kwargs):
        """Update client priority status."""
        try:
            body = json.loads(req.body)
            client_ip = body.get('client_ip')
            is_priority = body.get('is_priority', False)
            
            if not client_ip:
                return Response(status=400, body='Missing client_ip')
            
            success = self.fitness_center_app.update_client_priority(client_ip, is_priority)
            
            if success:
                return Response(status=200, body=json.dumps({'status': 'success'}))
            else:
                return Response(status=404, body='Client not found')
        except Exception as e:
            return Response(status=500, body=str(e))
    
    @route('fitness', '/fitness/config', methods=['GET'])
    def get_config(self, req, **kwargs):
        """Get current client configuration."""
        config = self.fitness_center_app.CLIENT_CONFIG
        body = json.dumps(config, indent=2)
        return Response(content_type='application/json', body=body)


# Application entry point information
app_manager.require_app('ryu.app.rest_topology')
app_manager.require_app('ryu.app.ws_topology')
