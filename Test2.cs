MCC.LoadPlugin(new Test());

//Script Extensions
 
public class Test : Plugin
{
    public override void Initialize()
    {
        Console.WriteLine("Привет " + Environment.UserName + " и YouGame.biz");
    }
}