import OriginExt as O


# app = O.Application()
# app.Visible = 1

Oapp = O.Application()
name = Oapp.CreatePage(Oapp.OPT_WORKSHEET, "MyBook", "Origin")
wbook = Oapp.Pages(name)
wks = wbook.Layers(0)
for ii in range(0, wks.Columns.Count, 1):
    tmp = range(0, 10)
    tmp = [x + ii for x in tmp]
    col = wks.Columns(ii)
    col.SetData(tmp)
Oapp.Save(r'C:\Users\Li Hang\Desktop\junk.opju')
Oapp.Exit()
# need to do this to make sure Origin is closed
del Oapp
