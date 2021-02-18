import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import pdb
import os

xodr_root = 'xodr_files/'
available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']

link_types = ['predecessor', 'successor']

for map_name in available_maps:

    xodr_fname = xodr_root + map_name + ".xodr"
    print(xodr_fname, ' has loaded.')
    
    with open(xodr_fname) as fi:
        xml_tree = ET.parse(fi)
    
    parent_map = {c: p for p in xml_tree.iter() for c in p}
    root_f = xml_tree.getroot()

    root_t = ET.Element(map_name)

    for road_f in root_f.iter('road'):
    #     pdb.set_trace()

        road_t = ET.SubElement(root_t, "road")
        road_t.set('id', road_f.attrib['id'])
        road_t.set('junction', road_f.attrib['junction'])

        lane_t = ET.SubElement(road_t, "lane")
        lane_ids = []
        lane_widths =[]

        for lane_f in road_f.iter('lane'):
            if lane_f.attrib['id'] == '0':
                continue

            lane_ids.append(lane_f.attrib['id'])
            for width_f in lane_f.iter('width'):
                lane_widths.append(float(width_f.attrib['a']))
                if (float(width_f.attrib['b']) !=0 
                    or float(width_f.attrib['c']) !=0 
                    or float(width_f.attrib['d']) !=0):
                    print('lane marking is not constant function') # means width of road changes over d
                    print(road_f.attrib)
#                     print(lane_f.attrib)
#                     print(width_f.attrib)
                break

        lane_t.set('ids', lane_ids)
        lane_t.set('widths', lane_widths)

        for link_type in link_types:
            lane_id_from=[]
            lane_id_to=[]
            for pre_f in road_f.iter(link_type):
                if 'elementType' in pre_f.attrib.keys():
                    pre_t = ET.SubElement(road_t, link_type)
                    pre_t.set('elementType', pre_f.attrib['elementType'])
                    pre_t.set('elementId', pre_f.attrib['elementId'])
                else:
                    lane_id_to.append(pre_f.attrib['id'])
                    lane_id_from.append(parent_map[parent_map[pre_f]].attrib['id'])
            pre_t.set('lane_id_from', lane_id_from)
            pre_t.set('lane_id_to', lane_id_to)

    for junction_f in root_f.iter('junction'):
        junction_t = ET.SubElement(root_t, "junction")
        junction_t.set('id', junction_f.attrib['id'])

        for connection_f in junction_f.iter('connection'):
            connection_t = ET.SubElement(junction_t, "connection")
            connection_t.set('id', connection_f.attrib['id'])
            connection_t.set('incomingRoad', connection_f.attrib['incomingRoad'])
            connection_t.set('connectingRoad', connection_f.attrib['connectingRoad'])
            connection_t.set('contactPoint', connection_f.attrib['contactPoint'])

            for laneLink_f in connection_f.iter('laneLink'):
                laneLink_t = ET.SubElement(connection_t, "laneLink")
                laneLink_t.set('from', laneLink_f.attrib['from'])
                laneLink_t.set('to', laneLink_f.attrib['to'])

    v_tree = ET.ElementTree(root_t)
    
    fname_light=  xodr_root + map_name+"_light.xodr"
    v_tree.write(fname_light, encoding="utf-8", xml_declaration=True)
    dom = xml.dom.minidom.parse(fname_light)
    
    pretty_xml_as_string = dom.toprettyxml()

    fname_pretty = xodr_root + map_name+"_light_pretty.xodr"
    with open(fname_pretty, "w") as files:
        files.write(pretty_xml_as_string)
    print(fname_pretty, ' has saved.')
    os.remove(fname_light)
