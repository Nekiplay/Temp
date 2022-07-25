MCC.LoadMacros(new Test());

//Script Extensions
 
public class Test : Macros
{
  public override bool OnKeyDown(Key key, bool repeat)
  {
    if (key == Key.R)
    {
      return true;
    }
    return false;
  }
}
