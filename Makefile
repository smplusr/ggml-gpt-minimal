SRCDIR	:= src/
OUTDIR	:= bin/
TARGET	:= $(OUTDIR)gpt-2
PARAMS	:= 




all:	$(TARGET)

run:	$(TARGET)
	cd $(OUTDIR) && ./gpt-2 $(PARAMS)

clean:
	$(RM) $(TARGET)


$(TARGET):
	cd $(SRCDIR) && $(MAKE)
	mv $(SRCDIR)/*.elf $(TARGET)
