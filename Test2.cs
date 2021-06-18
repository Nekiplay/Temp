MCC.LoadPlugin(new Test());

//Script Extensions
 
public class Test : Plugin
{
    public override void Initialize()
    {
        Console.WriteLine("Лол а как я тут оказался");
    }
}