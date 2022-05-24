


start: 
	watchmedo auto-restart \
	--directory=. \
	--pattern=*.py \
	--interval=5 \
	--recursive  \
	-- python start.py