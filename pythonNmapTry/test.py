import nmap
nm=nmap.PortScanner()
nm.scan('127.0.0.1','1-1000')
for host in nm.all_hosts():
	print('--------------------------------------')
	print('Host: %s (%s)'%(host,nm[host].hostname()))
	print('State: %s'%nm[host].state())
for proto in nm[host].all_protocols():
	print('--------------------------------------')
	print('Protocol: %s'%proto)
	lport=nm[host][proto].keys()
	lport.sort()
	for port in lport:
		print('Port: %s\tstate: %s'%(port,nm[host][proto][port]['state']))
