"""Per-system LogHub log formats and header->regex conversion."""
import re

# Canonical logparser format strings for the core labeled systems.
LOG_FORMATS = {
    "HDFS":        "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "Hadoop":      "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
    "Spark":       "<Date> <Time> <Level> <Component>: <Content>",
    "Zookeeper":   "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
    "BGL":         "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
    "HPC":         "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
    "Thunderbird": "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>",
}
