import re
import json

EMPTY_PREFIX  = '      -'
CHANGE_PREFIX = 'CHANGE-'

def extract_change_items( pmd_item ):
    output = []

    index = 0
    while( index < len( pmd_item ) ):
        name, value = pmd_item[ index ]
        if( name[ : len( CHANGE_PREFIX ) ] == CHANGE_PREFIX ):
            item = []
            item.append( (name, value) )

            index += 1
            if (index >= len(pmd_item)):
                output.append(item)
                break

            name, value = pmd_item[index]

            while( ( index < len( pmd_item ) ) and ( name[ : len( CHANGE_PREFIX ) ] != CHANGE_PREFIX )):
                item.append( (name, value) )

                index += 1
                if(index >= len( pmd_item )):
                    break
                name, value = pmd_item[index]
                #print name

            #print item
            #print "---"
            output.append( item )

        else:
            index += 1

    return output

def process_change_item( item, type_name, type_count, property_name ):
    name, value = item[0]
    #print "type_name = %s, type_count = %u, property_name = %s" % (type_name, type_count, property_name)
    #print name
    if( name != type_name ):
        #print "---"
        return False
    current_type_count = 1

    has_property = False
    index = 1
    while( index < len( item ) ):
        name, value = item[ index ]
        #print name
        if( name[ : len(EMPTY_PREFIX)] == EMPTY_PREFIX ):
            if( name[ len(EMPTY_PREFIX) : ] != type_name[ len( EMPTY_PREFIX ) : ] ):
                #print "---"
                return False
            else:
                current_type_count += 1

        elif( name == property_name ):
            has_property = True

        index += 1

    #print current_type_count
    #print "%s, %s" % (str(has_property), str(current_type_count == type_count))
    #print "---"
    return has_property and (current_type_count == type_count)


change_list = [
('CHANGE-POINT',1),
('CHANGE-POINT',2),
('CHANGE-DELETE',1),
('CHANGE-INSERT',1)]

property_list = ['FUNCTION', 'STRUCTURE', 'STABILITY']
#handle = open( 'test2.result-thermal-all', 'w' )

pmd = json.loads( open( 'pmd07Mar26.json', 'rb' ).read() )

count = 0
max_count = 0

output = []

change_items = []
for pmd_item in pmd:
    change_items.extend( extract_change_items( pmd_item ) )

result = {}
for change_item in change_items:
    for type_name, type_count in change_list:
        for property_name in property_list:
            name = "%s %u %s" % (type_name, type_count, property_name)
            if(not result.has_key(name)):
                result[name] = 0

            if(process_change_item(change_item, type_name, type_count, property_name)):
                result[name] += 1

for key, value in result.iteritems():
    print "%s -> %s" % (key, value)

#print change_items
