#!/usr/bin/env python
# coding: utf-8

import os
from pprint import pprint
from RISparser import readris
filepath = '/Users/valery/Desktop/citation/42.ris'
with open(filepath, 'r') as bibliography_file:
    entries = readris(bibliography_file)
    for entry in entries:
        print(entry)
        gost = ''
        if 'authors' in entry:
            authors = []
            for author in entry['authors']:
                author_sep = author.split(',')
                if len(author_sep) == 1:
                    authors.append(author_sep[0] + '.')
                else:
                    full_name = author_sep[1].split(' ')
                    if len(full_name) > 2:
                        authors.append(author_sep[0] + ' ' + full_name[1][0] + '.'+ full_name[2])
                    else:
                        authors.append(author_sep[0] + ' ' + full_name[1][0] + '.')
            gost += ", ".join(authors) + ' '
        else:
            if 'first_authors' in entry:
                authors = []
                for author in entry['first_authors']:
                    author_sep = author.split(',')
                    if len(author_sep) == 1:
                        authors.append(author_sep[0] + '.')
                    else:
                        full_name = author_sep[1].split(' ')
                        if len(full_name) > 2:
                            authors.append(author_sep[0] + ' ' + full_name[1][0] + '.'+ full_name[2])
                        else:
                            authors.append(author_sep[0] + ' ' + full_name[1][0] + '.')
                gost += ", ".join(authors) + ' '
            else:
                print('no author')
        if 'title' in entry:
            gost += entry['title'] + '//'
        else:
            if 'primary_title' in entry:
                gost += entry['primary_title'] + '//'
            else:
                print('no title and no primary title')
        if 'journal_name' in entry:
            gost += entry['journal_name']
        else:
            if 'alternate_title1' in entry:
                gost += entry['alternate_title1']
            elif 'alternate_title3' in entry:
                gost += entry['alternate_title3']
            else:
                print('no journal name and no alternate title1')
        if 'year' in entry:
            gost += ' - ' + entry['year']+ '.'
        else:
            if 'publication_year' in entry:
                date = entry['publication_year'].split('/')
                gost += ' - ' + date[0]+ '.'
            else:
                print('no year')
        if 'volume' in entry:
            gost += ' - Vol. ' + entry['volume'] + ','
        else:
            print('no volume')
        if 'number' in entry:
            gost += ' No. ' + entry['number'] + '.'
        else:
            print('no number')
        if 'start_page' in entry:
            gost += ' P. ' + entry['start_page']
        else:
            print('no start')
        if 'end_page' in entry:
            gost += ' - ' + entry['end_page'] + '.'
        else:
            gost += '.'
            print('no end')
        
        print(gost)




